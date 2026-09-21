# Architectural Decisions

Tài liệu này ghi các quyết định nền tảng để tránh codebase V2 quay lại các vấn đề cũ.

---

## ADR-001 — Persistent identity is not tracker identity

### Decision

`ObjectMemory.object_id` là persistent identity.

`tracker_id` chỉ là observation metadata ngắn hạn.

### Reason

Tracker có thể dropout hoặc đổi ID sau occlusion.

### Consequence

EventEngine không được phụ thuộc trực tiếp vào `tracker_id`.

---

## ADR-002 — Event logic uses seconds, not frame counts

### Decision

Mọi timeout/grace/confirmation dùng seconds.

### Reason

Runtime FPS có thể thay đổi và semantic refresh FPS có thể khác detector FPS.

### Consequence

Business logic phải dựa trên timestamp.

---

## ADR-003 — Evaluator first

### Decision

Xây evaluator trước khi tuning pipeline.

### Reason

Không có evaluator đáng tin cậy thì không biết thay đổi nào thực sự cải thiện hệ thống.

---

## ADR-004 — Domain layer is framework-independent

### Decision

Domain layer không import torch/ultralytics/cv2.

### Reason

Tách business semantics khỏi implementation model.

---

## ADR-005 — Event-level evaluation is primary

### Decision

Precision/Recall/F1 chính được tính ở event-level.

### Reason

Một event kéo dài nhiều frame không được biến thành nhiều TP.

---

## ADR-006 — Locked test set

### Decision

Test set không được dùng để tuning threshold/model.

### Reason

Tránh overfitting và metric không đáng tin.

---

## ADR-007 — Start simple, add models only after error analysis

### Decision

MVP bắt đầu:

- YOLO pretrained;
- ByteTrack;
- DINOv2 frozen;
- custom association + memory + FSM.

### Reason

Dataset khoảng 40 video; bottleneck hiện tại có khả năng lớn nằm ở identity/event logic hơn là model capacity.

### Consequence

Không fine-tune DINO/SAM/video transformer trước khi có evidence từ error analysis.

---

## ADR-008 — One experiment, one main variable

### Decision

Mỗi experiment chỉ đổi một nhóm biến chính.

### Reason

Cho phép attribution improvement/regression.

---

## ADR-009 — Scene anomalies are first-class state

### Decision

Lighting/global change/camera shift phải có `SceneStatus`, không xử lý rải rác bằng if statements.

### Reason

EventEngine cần guard thống nhất để tránh burst FP.

---

## ADR-010 — Outputs are product features

### Decision

JSONL, snapshots, annotated video và logs là first-class outputs.

### Reason

Chúng là bằng chứng để QA/debug/benchmark, không phải chỉ tiện ích phụ.
