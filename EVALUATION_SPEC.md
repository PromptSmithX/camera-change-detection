# Evaluation Specification

## 1. Goal

Đánh giá hệ thống ở **event-level**, không chỉ frame-level.

Metric phải trả lời đúng câu hỏi:

> System có phát hiện đúng event cần cảnh báo hay không?

---

## 2. Primary metrics

Cho từng event type và overall:

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2PR / (P + R)
```

Report tối thiểu:

- TP
- FP
- FN
- Precision
- Recall
- F1
- event latency nếu có

TN không phải metric trọng tâm ở event-level nhưng có thể giữ cho compatibility với baseline cũ nếu evaluator hiện tại cần.

---

## 3. Why event-level

Ví dụ ground truth event tồn tại từ 12s đến 40s và system confirm ở 14.2s.

Đó vẫn là một event detection hợp lệ nếu nằm trong tolerance.

Không được tính thành hàng trăm TP chỉ vì event kéo dài hàng trăm frame.

---

## 4. Event matching

Prediction `P` match Ground Truth `G` khi thỏa:

1. `P.type == G.type`;
2. temporal overlap hoặc start-time tolerance hợp lệ;
3. spatial/object identity condition hợp lệ;
4. một prediction chỉ match tối đa một GT;
5. một GT chỉ match tối đa một prediction.

Assignment dùng greedy có kiểm soát hoặc Hungarian trên match score.

---

## 5. Temporal criteria

Config đề xuất:

```yaml
evaluation:
  start_tolerance_seconds: 3.0
  min_temporal_overlap: 0.1
```

Nếu event là persistent/open-ended, có thể match dựa trên start/confirmation time và spatial/object criteria.

---

## 6. Spatial criteria

### FORGOTTEN_OBJECT

Có thể dùng:

- bbox IoU tại confirmation;
- centroid distance normalized theo ROI;
- optional class compatibility.

### MOVED_OBJECT

Cần xem xét cả:

- baseline location;
- `movement_outcome`;
- new location khi outcome là `relocated`;
- same object identity trong annotation.

Với `relocated`, prediction chỉ match nếu baseline/new location đủ tương thích theo configured criteria. Với `left_scene`, evaluator so khớp baseline location và outcome; `new_bbox` phải là `null`. Metrics vẫn được tính vào `MOVED_OBJECT`, đồng thời report riêng cho `relocated` và `left_scene`.

---

## 7. Duplicate prediction handling

Nếu một GT có 3 prediction tương tự:

- 1 prediction tốt nhất → TP;
- 2 prediction còn lại → FP duplicate.

Điều này bắt buộc để chống duplicate events.

---

## 8. Error taxonomy

Mỗi FP/FN nên được gắn nguyên nhân nếu có thể:

### FP categories

- lighting/global change;
- person occlusion;
- tracker ID switch;
- detector false positive;
- association mismatch;
- duplicate event;
- short-lived object;
- ROI edge noise;
- unknown.

### FN categories

- detector miss;
- encoder/appearance failure;
- association failure;
- threshold too strict;
- timeout too long;
- scene anomaly freeze quá lâu;
- object too small;
- unknown.

---

## 9. Benchmark output

`metrics.json`:

```json
{
  "overall": {
    "tp": 0,
    "fp": 0,
    "fn": 0,
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0
  },
  "FORGOTTEN_OBJECT": {},
  "MOVED_OBJECT": {},
  "MOVED_OBJECT_OUTCOMES": {
    "relocated": {},
    "left_scene": {}
  },
  "error_breakdown": {}
}
```

Console summary:

```text
Overall
  Precision: ...
  Recall:    ...
  F1:        ...

Forgotten
  Precision: ...
  Recall:    ...
  F1:        ...

Moved
  Precision: ...
  Recall:    ...
  F1:        ...
```

---

## 10. Benchmark discipline

### Validation

Được dùng để:

- tune threshold;
- tune association weights;
- chọn model/tracker;
- tune temporal durations.

### Locked test

Chỉ chạy sau khi:

- logic đã freeze;
- threshold đã freeze;
- model/config đã freeze.

Không tuning lại dựa trên test result.

---

## 11. Current baseline reference

Baseline hệ thống cũ:

- Precision: 0.4444
- Recall: 0.2667
- F1: 0.3333
- FORGOTTEN_OBJECT F1: 0.3810
- MOVED_OBJECT F1: 0.0

V2 phải giữ baseline này để so sánh regression/improvement, nhưng không được coi nó là target architecture.
