# Repository Structure & Coding Boundaries

## 1. Proposed repository

```text
change-detection-v2/
├── pyproject.toml
├── README.md
├── configs/
│   ├── default.yaml
│   ├── debug.yaml
│   └── production.yaml
├── docs/
│   └── ...
├── src/
│   └── change_detection/
│       ├── domain/
│       │   ├── entities.py
│       │   ├── enums.py
│       │   ├── events.py
│       │   └── geometry.py
│       ├── sources/
│       │   ├── base.py
│       │   ├── video.py
│       │   └── webcam.py
│       ├── scene/
│       │   ├── roi.py
│       │   ├── calibration.py
│       │   ├── baseline.py
│       │   └── scene_monitor.py
│       ├── perception/
│       │   ├── detector.py
│       │   ├── tracker.py
│       │   ├── encoder.py
│       │   └── observation_builder.py
│       ├── association/
│       │   ├── matcher.py
│       │   ├── similarity.py
│       │   └── assignment.py
│       ├── memory/
│       │   ├── object_memory.py
│       │   └── memory_object.py
│       ├── events/
│       │   ├── engine.py
│       │   ├── forgotten.py
│       │   ├── moved.py
│       │   └── state_machine.py
│       ├── output/
│       │   ├── event_store.py
│       │   ├── jsonl_writer.py
│       │   ├── snapshot_writer.py
│       │   └── video_writer.py
│       ├── evaluation/
│       │   ├── evaluator.py
│       │   ├── event_matching.py
│       │   ├── metrics.py
│       │   └── report.py
│       ├── pipeline/
│       │   ├── pipeline.py
│       │   └── runner.py
│       └── infrastructure/
│           ├── models/
│           │   ├── yolo.py
│           │   └── dino.py
│           ├── tracking/
│           │   ├── bytetrack.py
│           │   └── botsort.py
│           └── logging.py
├── tools/
│   ├── annotate.py
│   ├── validate_dataset.py
│   ├── run_video.py
│   ├── benchmark.py
│   └── visualize_results.py
└── tests/
    ├── unit/
    ├── integration/
    └── scenarios/
```

---

## 2. Dependency rule

Allowed dependency direction:

```text
infrastructure ──► domain interfaces
application/pipeline ──► domain + interfaces
models/frameworks ──► infrastructure adapters only
```

Forbidden:

```text
domain ─X─► ultralytics
 domain ─X─► torch
 domain ─X─► cv2
 events ─X─► YOLO result object
```

---

## 3. Model adapters

Framework-specific conversion phải xảy ra ở adapter boundary.

Ví dụ:

```python
# infrastructure/models/yolo.py

def detect(...) -> list[Detection]:
    raw_results = yolo(...)
    return convert_to_domain_detections(raw_results)
```

Code bên ngoài không được truyền raw Ultralytics result.

---

## 4. Configuration

Không hardcode threshold trong Python.

The current M0/M1 implementation loads JSON and TOML. The YAML-shaped
example below remains an architectural illustration until a YAML loader is
explicitly added.

Example:

```yaml
runtime:
  processing_fps: 5

perception:
  detector:
    name: yolo
    confidence: 0.35
  encoder:
    name: dinov2
    semantic_refresh_fps: 2.5

association:
  appearance_weight: 0.50
  spatial_weight: 0.20
  size_weight: 0.15
  class_weight: 0.15

events:
  forgotten:
    candidate_seconds: 1.0
    confirm_seconds: 4.0
  moved:
    missing_grace_seconds: 1.0
    confirm_seconds: 2.0

occlusion:
  grace_seconds: 2.0
```

---

## 5. Time convention

Business logic dùng `timestamp_sec` / duration in seconds.

Không dùng logic kiểu:

```python
if missing_frames > 20:
```

Thay bằng:

```python
if missing_duration_sec >= config.missing_grace_seconds:
```

---

## 6. IDs

- `video_id`: stable dataset/runtime video ID
- `camera_id`: stable camera ID
- `tracker_id`: temporary, unreliable
- `object_id`: persistent identity trong ObjectMemory
- `event_id`: event lifecycle ID

Không dùng lẫn các ID.

---

## 7. Logging

Structured logging fields khuyến nghị:

```text
video_id
frame_index
timestamp_sec
tracker_id
object_id
state_before
state_after
association_score
event_id
event_action
```

---

## 8. Run artifacts

```text
runs/<run_id>/
├── config.yaml
├── metadata.json
├── events.jsonl
├── metrics.json
├── snapshots/
├── video/
└── logs/
```

---

## 9. Pull request rule

Mỗi PR thay đổi event semantics phải:

- update EVENT_SPEC.md;
- thêm/chỉnh scenario tests;
- chạy benchmark validation;
- ghi metric before/after.

Mỗi PR đổi model adapter nhưng không đổi semantics không được sửa event definition để “fit” model output.
