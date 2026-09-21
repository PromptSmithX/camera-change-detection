# Event Specification

## 1. Purpose

File này định nghĩa chính xác semantics và lifecycle của các event trong hệ thống V2.

Event engine phải deterministic và test được không cần model/GPU.

---

## 2. Shared concepts

### Baseline object

Object tồn tại trong scene tại thời điểm calibration và được lưu trong `SceneBaseline`.

### New object

Observation/object xuất hiện sau calibration nhưng không match hợp lệ với baseline/memory object hiện có.

### Persistent identity

Persistent identity do `ObjectMemory.object_id` quản lý.

`tracker_id` chỉ là tín hiệu ngắn hạn và có thể thay đổi.

### Stable

Object được xem là stable khi thỏa các điều kiện configured, ví dụ:

- tồn tại liên tục đủ thời gian;
- centroid displacement nhỏ;
- bbox/area không dao động quá lớn;
- không nằm trong scene anomaly.

---

## 3. FORGOTTEN_OBJECT

### 3.1 Definition

`FORGOTTEN_OBJECT` là object:

1. không thuộc baseline;
2. xuất hiện trong ROI;
3. không match persistent identity hiện hữu;
4. ổn định đủ thời gian;
5. không bị giải thích bởi short occlusion, tracking artifact hoặc global scene anomaly;
6. chưa có active forgotten event cho cùng object.

### 3.2 State machine

```text
UNKNOWN
   │ first seen
   ▼
NEW
   │ stable >= candidate_seconds
   ▼
CANDIDATE
   │ persistent >= confirm_seconds
   ▼
CONFIRMED
   │ disappears / resolved
   ▼
CLOSED
```

### 3.3 Guards

Không tăng confirmation timer khi:

- scene lighting anomaly active;
- camera shift active;
- object bị occluded vượt configured overlap condition;
- observation confidence quá thấp;
- object nằm ngoài ROI hợp lệ.

### 3.4 Deduplication

Một `MemoryObject.object_id` chỉ có tối đa một active `FORGOTTEN_OBJECT` event.

### 3.5 Suggested config

```yaml
events:
  forgotten:
    candidate_seconds: 1.0
    confirm_seconds: 4.0
    disappear_grace_seconds: 1.0
```

Các giá trị chỉ là starting point, phải tune trên validation.

---

## 4. MOVED_OBJECT

### 4.1 Definition

`MOVED_OBJECT` xảy ra khi một baseline object được xác nhận có cùng identity nhưng ở vị trí mới đủ xa vị trí baseline.

Phải có bằng chứng đồng thời:

- baseline object absent tại old location;
- new/current object xuất hiện tại new location;
- appearance similarity đủ cao;
- class/geometry tương thích nếu available;
- displacement vượt threshold;
- trạng thái ổn định đủ lâu.

### 4.2 Không được suy luận moved chỉ từ

```text
missing baseline object + arbitrary new object
```

Phải có identity association.

### 4.3 State machine

```text
BASELINE_PRESENT
       │ absent
       ▼
TEMP_MISSING
       │
       ├── reappears old location ──► BASELINE_PRESENT
       │
       └── matching identity at new location
                          │
                          ▼
                    MOVE_CANDIDATE
                          │ stable >= confirm_seconds
                          ▼
                    MOVED_CONFIRMED
                          │ resolved / recalibrated
                          ▼
                        CLOSED
```

### 4.4 Identity score

Initial conceptual score:

```text
identity_score =
    w_a * appearance_similarity
  + w_s * size_similarity
  + w_c * class_compatibility
  + w_t * temporal_consistency
```

Spatial position không nên được dùng theo cách làm giảm điểm mạnh chỉ vì object đã di chuyển xa; spatial signal nên dùng chủ yếu để:

- xác nhận old position đã trống;
- xác nhận displacement đủ lớn;
- loại candidate bất hợp lý.

### 4.5 Suggested config

```yaml
events:
  moved:
    missing_grace_seconds: 1.0
    confirm_seconds: 2.0
    min_identity_score: 0.75
    min_displacement_ratio: 0.05
```

---

## 5. Occlusion behavior

Khi object bị person/foreground object che:

```text
PRESENT → OCCLUDED → PRESENT
```

Trong thời gian occlusion hợp lệ:

- không tăng missing timer;
- không tạo MOVED candidate;
- không close active event;
- tracker ID có thể thay đổi sau occlusion mà không ảnh hưởng persistent identity.

---

## 6. Tracker dropout behavior

Short dropout:

```text
PRESENT
→ TEMP_MISSING
→ PRESENT
```

Nếu object quay lại trong grace window và association score đủ cao:

- giữ nguyên `object_id`;
- không tạo new object;
- không tạo duplicate event.

---

## 7. Scene anomaly behavior

Khi `SceneStatus` cho biết global lighting change hoặc camera shift:

- freeze candidate confirmation timer;
- không tạo burst new objects;
- có thể cho phép ObjectMemory tiếp tục giữ last-known state;
- chỉ resume event logic khi scene stable lại.

---

## 8. Event lifecycle

Mỗi event có lifecycle:

```text
CREATED → ACTIVE → CLOSED
```

Schema đề xuất:

```python
@dataclass
class Event:
    event_id: UUID
    event_type: EventType
    object_id: UUID
    started_at_sec: float
    confirmed_at_sec: float
    ended_at_sec: float | None
    confidence: float
    before_bbox: BBox | None
    after_bbox: BBox | None
```

---

## 9. Event output contract

JSONL public event example:

```json
{
  "event_id": "evt_001",
  "type": "MOVED_OBJECT",
  "object_id": "obj_005",
  "started_at_sec": 28.4,
  "confirmed_at_sec": 30.4,
  "ended_at_sec": null,
  "confidence": 0.87,
  "from_bbox": [100, 150, 170, 260],
  "to_bbox": [420, 155, 490, 265]
}
```

Không đưa private/internal state không cần thiết vào public JSONL.
