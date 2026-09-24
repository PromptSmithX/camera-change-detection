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

- [x] Tạo repo + pyproject.
- [x] Tạo domain dataclasses/enums.
- [x] Tạo config loading/validation.
- [x] Chuẩn hóa dataset manifest.
- [x] Chuẩn hóa annotation schema.
- [x] Viết `validate_dataset.py`.
- [x] Viết event matcher.
- [x] Viết Precision/Recall/F1 evaluator.
- [x] Tạo baseline metrics record từ hệ thống cũ.

## Deliverables

```text
tools/validate_dataset.py
tools/benchmark.py
src/.../domain/
src/.../evaluation/
```

## Exit criteria

- [x] Dataset hiện có pass validator.
- [x] Evaluator chạy với fake predictions.
- [x] Unit tests evaluator pass.
- [x] Locked test split được định nghĩa.

---

# M1 — Source, ROI & Calibration

## Tasks

- [x] `FrameSource` protocol.
- [x] `VideoSource`.
- [x] `ImageSequenceSource` for the canonical benchmark media type.
- [x] `WebcamSource`.
- [x] ROI model + clipping.
- [x] Scene stability measurement.
- [x] Calibration frame collection.
- [x] Reference scene creation.
- [x] `SceneBaseline` persistence.

## Exit criteria

- [x] Video/image sequence đọc được frame/timestamp đúng; webcam có monotonic timestamps.
- [x] ROI hoạt động với BBox typed format và legacy ROI JSON.
- [x] Scene moving → calibration fail.
- [x] Scene stable → baseline được save/load với checksum.

---

# M2 — Detector & Short-term Tracking

## Initial model choice

- Detector: YOLO small pretrained.
- Tracker: ByteTrack.

## Tasks

- [x] Detector interface.
- [x] YOLO adapter.
- [x] Tracker interface.
- [x] ByteTrack adapter.
- [x] Observation builder.
- [x] Visual debug overlay.

## Exit criteria

- [x] Pipeline hiển thị bbox, class, confidence.
- [x] Track continuity hoạt động basic.
- [x] Model-specific result không leak ra domain layer.

---

# M3 — Feature Encoder, Association & ObjectMemory

## Initial choice

- Encoder: DINOv2 frozen.

## Tasks

- [x] Crop policy + padding.
- [x] Encoder interface.
- [x] DINOv2 adapter.
- [x] Appearance similarity.
- [x] Spatial/size/class similarity.
- [x] Gating.
- [x] Assignment matching.
- [x] ObjectMemory implementation.
- [x] Observation history.

## Exit criteria

- [x] Same object giữ `object_id` dù `tracker_id` đổi trong deterministic test.
- [x] Short dropout không tạo object mới.
- [x] Association debug log có score breakdown.

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

## M0 implementation status (2026-09-22) — COMPLETED ✅

All M0 deliverables are complete:

- Domain contracts, config, enums (`src/change_detection/domain/`).
- Dataset validator (`tools/validate_dataset.py`) — draft and official modes.
- Event-level evaluator (`tools/benchmark.py`) with bipartite matching.
- Legacy baseline metrics record (`data/benchmark/v1/baseline_legacy.json`).
- Local review tool (`tools/annotate.py`) and apply-reviews pipeline.
- 22/22 unit tests pass.

Dataset gate passed:

- 44 review files manually verified with `confirmation_time_sec` and ROI.
- Reviews applied to produce `data/benchmark/v2/` (version `1.0.0`).
- 2 MEVA samples excluded (no source review available).
- Official validation: **0 errors, 0 warnings** (`--mode official --strict-hashes`).
- Test split locked: 21 test + 23 validation = 44 active samples.
- Lock artifacts: `data/benchmark/v2/locked_test.json`, `data/benchmark/v2/manifest.locked.json`.

M1 (Source, ROI & Calibration) may now begin.

## M1 implementation status (2026-09-22) — COMPLETED

M1 runtime foundations are implemented:

- `VideoSource`, `ImageSequenceSource`, and `WebcamSource` share the `FrameSource` contract.
- BBox ROI clipping supports typed M1 records and existing benchmark ROI files.
- Scene stability, deterministic calibration, and `SceneBaseline` persistence are covered by tests.
- Baseline artifacts are written under `runs/`; benchmark and locked-test files remain untouched.
- Detector, tracker, encoder, and baseline object recognition remain scoped to M2.

## M2 implementation status (2026-09-22) — IMPLEMENTED

M2 perception code and deterministic tests are implemented:

- Framework-independent `Detection`, `Track`, and `Observation` contracts.
- Ultralytics YOLO adapter with ROI crop/coordinate translation.
- Supervision ByteTrack adapter with short-term tracker reset and ID mapping.
- Reusable source/ROI construction and `tools/run_perception.py` runner.
- JSONL observation records, metadata, and annotated MP4 debug output.
- Optional calibration-time baseline detections; persistent identity remains deferred to M3.

The local unit/integration suite passes. A real YOLO/ByteTrack smoke test was
run with `weights/yolov8s.pt` on the RTX 3050 CUDA device: 900 processed
frames, 627 detections, and 462 short-term tracks. Artifacts are under
`runs/m2/rtx3050/`; automatic GitHub weight download is not part of the
repository workflow.

## M3 implementation status (2026-09-23) — IMPLEMENTED

M3 adds frozen DINOv2 ViT-S/14 embeddings, deterministic Hungarian association,
and bounded persistent object memory without changing the M2 observation
artifact schema. The deterministic suite covers tracker ID switch, short
dropout, stale transitions, DINO crop/preprocess behavior, M3 baseline schema
v2, and identity debug artifacts. The real GPU smoke test passed with the
official DINOv2 checkpoint cached in `runs/torch-hub/`: 900 frames processed,
627 detections, 462 observations, and 462 persistent identity assignments.
Artifacts are under `runs/m3/baseline/` and `runs/m3/smoke/`.
