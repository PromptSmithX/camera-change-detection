# Implementation Plan — Change Detection System V2

## 1. Strategy

Thứ tự triển khai cố định:

```text
Domain contracts
→ Dataset + Evaluator
→ Source + ROI + Calibration
→ Detector + Tracker
→ Encoder + Association
→ Object Memory
→ FORGOTTEN FSM
→ MOVED FSM
→ Robustness
→ Benchmark + Optimization
```

Không bắt đầu bằng fine-tune model.

---

## 2. Milestones overview

| Milestone | Mục tiêu | Exit condition |
|---|---|---|
| M0 | Foundation + evaluator | Dataset validate + evaluator chạy độc lập |
| M1 | Source / ROI / calibration | Tạo SceneBaseline reproducible |
| M2 | Detection / tracking | Visualize detections + short tracks |
| M3 | Encoder / association / memory | Persistent object ID hoạt động |
| M4 | FORGOTTEN_OBJECT | Scenario tests + validation metric |
| M5 | MOVED_OBJECT | Identity-based moved detection hoạt động |
| M6 | Robustness | Occlusion/light/dropout/ID-switch tests pass |
| M7 | Benchmark / optimize | Config locked + benchmark report |

---

# M0 — Foundation & Evaluator

## Tasks

- [ ] Tạo repo + pyproject.
- [ ] Tạo domain dataclasses/enums.
- [ ] Tạo config loading/validation.
- [ ] Chuẩn hóa dataset manifest.
- [ ] Chuẩn hóa annotation schema.
- [ ] Viết `validate_dataset.py`.
- [ ] Viết event matcher.
- [ ] Viết Precision/Recall/F1 evaluator.
- [ ] Tạo baseline metrics record từ hệ thống cũ.

## Deliverables

```text
tools/validate_dataset.py
tools/benchmark.py
src/.../domain/
src/.../evaluation/
```

## Exit criteria

- [ ] Dataset hiện có pass validator.
- [ ] Evaluator chạy với fake predictions.
- [ ] Unit tests evaluator pass.
- [ ] Locked test split được định nghĩa.

---

# M1 — Source, ROI & Calibration

## Tasks

- [ ] `FrameSource` protocol.
- [ ] `VideoSource`.
- [ ] `WebcamSource`.
- [ ] ROI model + clipping.
- [ ] Scene stability measurement.
- [ ] Calibration frame collection.
- [ ] Reference scene creation.
- [ ] `SceneBaseline` persistence.

## Exit criteria

- [ ] Video đọc được frame/timestamp đúng.
- [ ] ROI hoạt động.
- [ ] Scene moving → calibration fail.
- [ ] Scene stable → baseline được save/load.

---

# M2 — Detector & Short-term Tracking

## Initial model choice

- Detector: YOLO small pretrained.
- Tracker: ByteTrack.

## Tasks

- [ ] Detector interface.
- [ ] YOLO adapter.
- [ ] Tracker interface.
- [ ] ByteTrack adapter.
- [ ] Observation builder.
- [ ] Visual debug overlay.

## Exit criteria

- [ ] Pipeline hiển thị bbox, class, confidence.
- [ ] Track continuity hoạt động basic.
- [ ] Model-specific result không leak ra domain layer.

---

# M3 — Feature Encoder, Association & ObjectMemory

## Initial choice

- Encoder: DINOv2 frozen.

## Tasks

- [ ] Crop policy + padding.
- [ ] Encoder interface.
- [ ] DINOv2 adapter.
- [ ] Appearance similarity.
- [ ] Spatial/size/class similarity.
- [ ] Gating.
- [ ] Assignment matching.
- [ ] ObjectMemory implementation.
- [ ] Observation history.

## Exit criteria

- [ ] Same object giữ `object_id` dù `tracker_id` đổi trong deterministic test.
- [ ] Short dropout không tạo object mới.
- [ ] Association debug log có score breakdown.

---

# M4 — FORGOTTEN_OBJECT

## Tasks

- [ ] Forgotten FSM.
- [ ] candidate timer.
- [ ] stable condition.
- [ ] disappearance/close logic.
- [ ] duplicate prevention.
- [ ] EventStore integration.
- [ ] snapshot BEFORE/AFTER.

## Exit criteria

- [ ] New stable object → one forgotten event.
- [ ] Short-lived object → no event.
- [ ] Same object qua ID switch → no duplicate.
- [ ] Validation metric được sinh riêng cho FORGOTTEN_OBJECT.

---

# M5 — MOVED_OBJECT

Đây là milestone ưu tiên cao nhất vì baseline hiện tại có F1 MOVED_OBJECT = 0.

## Tasks

- [ ] Baseline object absence tracking.
- [ ] Candidate identity search tại new location.
- [ ] Identity scoring.
- [ ] Displacement rule.
- [ ] Move candidate state.
- [ ] Confirmation timer.
- [ ] from/to output.
- [ ] Reappearance old-location cancellation.

## Exit criteria

- [ ] Same object moved → moved event.
- [ ] Missing + unrelated new object → no moved event.
- [ ] Temporary disappearance → no moved event.
- [ ] Moved event output có from/to bbox.
- [ ] Validation metric riêng cho MOVED_OBJECT.

---

# M6 — Robustness

## Tasks

- [ ] Person/foreground occlusion detection.
- [ ] Occlusion grace logic.
- [ ] SceneMonitor global lighting change.
- [ ] Camera shift detection basic.
- [ ] Tracker dropout grace.
- [ ] ID switch regression tests.
- [ ] ROI edge noise filtering.
- [ ] Long-video memory test.

## Exit criteria

- [ ] Robustness scenario tests pass.
- [ ] Không burst FP khi lighting change.
- [ ] Không duplicate event do dropout/ID switch.
- [ ] Long video không corrupt output.

---

# M7 — Benchmark & Optimization

## Experiments

Chỉ đổi một biến mỗi experiment.

### E0

Legacy/current baseline.

### E1

V2 memory + FSM, chưa semantic improvement thêm.

### E2

E1 + DINO appearance memory.

### E3

E2 + scene/change proposal optimization.

### E4

BBox → segmentation experiment.

### Optional later

- ByteTrack vs BoT-SORT.
- DINOv2 vs DINOv3.
- YOLO detect vs YOLO-seg.
- SAM-based refinement cho ambiguous candidates.

## Exit criteria

- [ ] Validation config locked.
- [ ] Run locked test exactly once per release candidate policy.
- [ ] Report metric tổng + từng event type.
- [ ] Error taxonomy report.
- [ ] Performance report.

---

## 3. Weekly execution template

Mỗi chu kỳ dev:

```text
1. Chọn 1 hypothesis.
2. Viết/ cập nhật scenario test trước.
3. Implement thay đổi.
4. Chạy unit + integration.
5. Chạy validation benchmark.
6. So sánh before/after.
7. Ghi error analysis.
8. Giữ hoặc revert thay đổi.
```

---

## 4. Definition of ready before model tuning

Chỉ bắt đầu tuning/fine-tune model khi:

- evaluator tin cậy;
- annotation đủ tốt;
- object association debug được;
- event FSM pass deterministic tests;
- lỗi detector thực sự được chứng minh là bottleneck qua FN analysis.

---

## 5. Release checklist

- [ ] Config versioned.
- [ ] Dataset version recorded.
- [ ] Model versions recorded.
- [ ] All automated tests pass.
- [ ] Validation benchmark saved.
- [ ] No P0.
- [ ] No unresolved P1 affecting main event semantics.
- [ ] JSONL valid.
- [ ] Snapshot valid.
- [ ] Annotated video valid.
- [ ] Locked test untouched during tuning.

## M0 implementation status (2026-09-21)

The repository now contains the independent domain contracts, canonical draft
migration, draft/official validator, event-level evaluator, fake-prediction
tests, local review tool, QA report, and fail-closed test-lock command.

The current `data/benchmark/v1/` is intentionally **not** an official locked
benchmark yet: it contains 46 active samples (23 validation / 23 test), with
44 provisional samples and 2 excluded MEVA samples. Manual review must fill
confirmation timing, moved-object `new_bbox`, persistent identity, ROI and
reference decisions before running `lock_test.py`. Model pipeline work starts
only after that gate passes.
