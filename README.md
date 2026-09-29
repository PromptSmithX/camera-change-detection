# Change Detection System V2 — Documentation Index

Bộ tài liệu này là source of truth cho việc xây lại hệ thống phát hiện thay đổi trên camera cố định.

## Mục tiêu hệ thống

Hệ thống nhận video file hoặc webcam/camera cố định, cho phép chọn ROI, hiệu chuẩn baseline và phát hiện hai loại sự kiện chính:

- `FORGOTTEN_OBJECT`: vật thể mới xuất hiện trong ROI và tồn tại ổn định đủ lâu.
- `MOVED_OBJECT`: vật thể thuộc baseline bị dịch chuyển khỏi vị trí ban đầu, sang vị trí mới hoặc rời ROI/camera.

Hệ thống phải chịu được các tình huống phổ biến như thay đổi ánh sáng, người che khuất, tracker dropout, đổi track ID và nhiễu nhỏ gần biên ROI.

## Tài liệu

| File | Nội dung |
|---|---|
| [PRD.md](./PRD.md) | Product requirements, scope, KPI, user flow, acceptance criteria |
| [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md) | Kiến trúc tổng thể, module, data flow, interfaces |
| [EVENT_SPEC.md](./EVENT_SPEC.md) | Định nghĩa FORGOTTEN_OBJECT, MOVED_OBJECT, FSM và lifecycle |
| [DATASET_ANNOTATION_SPEC.md](./DATASET_ANNOTATION_SPEC.md) | Chuẩn dataset, annotation, split train/val/test |
| [EVALUATION_SPEC.md](./EVALUATION_SPEC.md) | Cách tính TP/FP/FN, Precision/Recall/F1, event matching |
| [REPO_STRUCTURE.md](./REPO_STRUCTURE.md) | Cấu trúc source code, dependency boundaries, conventions |
| [TEST_PLAN.md](./TEST_PLAN.md) | Test strategy, unit/integration/scenario/performance tests |
| [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) | Roadmap theo milestone từ M0 đến M7 |
| [DECISIONS.md](./DECISIONS.md) | Architectural decisions và nguyên tắc không được phá vỡ |

## Nguyên tắc cốt lõi

1. `object_id` của hệ thống **không phải** `tracker_id`.
2. Event được xác nhận bằng temporal logic theo **giây**, không hardcode theo frame.
3. Model chỉ là adapter/plugin; business logic không phụ thuộc trực tiếp Ultralytics/PyTorch/OpenCV.
4. Evaluator và dataset contract được xây trước khi tuning model.
5. Test set phải được khóa; không tuning threshold/model trên test set.
6. Mỗi experiment chỉ thay đổi một nhóm biến chính để có thể attribution improvement.

## Baseline hiện tại

Kết quả hệ thống cũ được dùng làm baseline ban đầu:

- Precision: `0.4444`
- Recall: `0.2667`
- F1: `0.3333`
- TP / FP / FN / TN: `4 / 5 / 11 / 7`
- F1 FORGOTTEN_OBJECT: `0.3810`
- F1 MOVED_OBJECT: `0.0`

Mục tiêu MVP ban đầu: xây được pipeline V2 có evaluation reproducible, sau đó cải thiện dần đến KPI nghiệm thu được chốt bởi team.

## M0 implementation quickstart

The canonical, non-destructive benchmark draft is generated under
`data/benchmark/v1/`; the original `data/processed/` files are not overwritten.

```text
python tools/migrate_dataset.py
python tools/validate_dataset.py --manifest data/benchmark/v1/manifest.json --root . --mode draft --strict-hashes
python tools/qa_report.py --manifest data/benchmark/v1/manifest.json --root .
python -m pytest
python tools/annotate.py --manifest data/benchmark/v1/manifest.json --root .
```

Use the validation split while reviewing and tuning. The annotation tool writes
drafts to `data/benchmark/v1/reviews/` and never loads predictions. After all
46 samples and 35 events are reviewed and official validation is clean, create the
read-only test artifacts with:

First materialize the review drafts as a new version (this leaves the current
draft untouched):

```text
python tools/apply_reviews.py --manifest data/benchmark/v1/manifest.json --reviews-dir data/benchmark/v1/reviews --output-dir data/benchmark/v1/reviewed --dataset-version 1.0.1-reviewed --root .
python tools/validate_dataset.py --manifest data/benchmark/v1/reviewed/manifest.json --root . --mode official --strict-hashes
```

```text
python tools/lock_test.py --manifest data/benchmark/v1/reviewed/manifest.json --root . --dataset-version 1.0.2-locked
```

`lock_test.py` fails closed while any sample/event is provisional, excluded,
incomplete, or has unresolved validator warnings.

## M1 calibration quickstart

M1 supports video files, image sequences, and webcams through the same source
contract. Configuration is currently JSON or TOML. The example below uses a
validation image sequence and writes only to `runs/`:

```text
python tools/calibrate.py --config configs/m1.example.json --output runs/m1-example/baseline
```

M1 creates a scene reference and stability summary. Detector-level baseline
records are added in M2; baseline embeddings and persistent identity start in
M3.

## M2 perception quickstart

Install the optional detector/tracker runtime into the project environment:

```text
python -m pip install -e ".[perception]"
```

For an NVIDIA CUDA environment, install the matching PyTorch wheel before the
perception extra so the detector uses the GPU instead of a CPU-only torch:

```text
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
```

Place `yolov8s.pt` at `weights/yolov8s.pt` (or change
`perception.detector.model`), keep `perception.detector.device` as `"0"` for
the first CUDA GPU, then run the detector and short-term tracker:

```text
python tools/run_perception.py --config configs/m2.example.json --output runs/m2/example
```

For CPU fallback, set `perception.detector.device` to `"cpu"`.

M2 writes `metadata.json`, `observations.jsonl`, and `annotated.mp4` under
`runs/`. It exposes only short-term `tracker_id`; persistent identity and event
logic start in M3/M4.

## M3 identity quickstart

M3 keeps the M2 runner and adds frozen DINOv2 appearance embeddings,
association, and persistent object memory. Install the identity extra after
installing the CUDA-compatible Torch wheel described above:

```text
python -m pip install -e ".[perception,identity]"
python tools/calibrate.py --config configs/m3.example.json --output runs/m3/baseline
python tools/run_perception.py --config configs/m3.example.json --output runs/m3/example
```

The first calibration downloads the official `dinov2_vits14` checkpoint to
`runs/torch-hub/`; later runs use the cache. M3 calibration writes a schema-v2
baseline containing persistent baseline IDs and embeddings. M3 runtime keeps
the M2 `observations.jsonl` contract and additionally writes `identity.jsonl`
with association candidates, score breakdowns, object IDs, and state
transitions. Raw embeddings are not written to run logs.

## M4/M5 event quickstart

M4/M5 consumes the schema-v2 baseline and adds the deterministic forgotten and
moved-object FSMs. The reference configuration is
`configs/m45.example.json`:

```text
python tools/run_change_detection.py \
  --config configs/m45.example.json \
  --output runs/m45/cdnet2014_abandonedbox
```

The run writes the M2/M3 artifacts plus `events.jsonl`,
`event_lifecycle.jsonl`, and BEFORE/AFTER images under `snapshots/`. Event
timers are measured in seconds; `tracker_id` is never used as the persistent
event identity.

Generate validation metrics per sample and per event type with:

```text
python tools/benchmark_dataset.py \
  --manifest data/benchmark/v2/manifest.json \
  --predictions-dir runs/m45 \
  --split validation \
  --output runs/m45/metrics.json
```

Run the complete validation split sequentially, generating a baseline from each
sample's annotated reference-frame range and preserving per-sample artifacts,
configs, and logs. The batch writes both the existing-compatible metrics and a
second report that only counts events confirmed no more than three seconds after
the ground-truth confirmation:

```text
python tools/run_validation.py --config configs/m45.example.json
```

The command creates a new timestamped directory under `runs/validation/`.
Pass `--output runs/validation/my-run` to choose a stable path; after an
interruption, use the same path with `--resume`. Resume is rejected if the
manifest, media, base config, evaluation settings, or pipeline code changed.

The test split is deliberately opt-in: use the locked manifest and pass
`--allow-test` explicitly.
