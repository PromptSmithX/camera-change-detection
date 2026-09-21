# System Architecture — Change Detection System V2

## 1. Architecture goals

Kiến trúc được thiết kế để tách rõ:

- perception từ event logic;
- temporary tracking từ persistent object identity;
- runtime pipeline từ evaluation;
- model adapters từ domain logic.

Nguyên tắc: **model có thể thay, event semantics không được thay theo model**.

---

## 2. High-level architecture

```text
Video / Webcam
      │
      ▼
┌──────────────────┐
│ FrameSource      │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Preprocess / ROI │
└────────┬─────────┘
         ├──────────────────► SceneMonitor
         │
         ▼
┌──────────────────┐
│ Detector         │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Short Tracker    │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Feature Encoder  │
└────────┬─────────┘
         ▼
┌──────────────────────────┐
│ Observation Builder      │
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ Association Engine       │
│ appearance + spatial +   │
│ geometry + class         │
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ Object Memory            │
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ Event Engine / FSM       │
│ - Forgotten             │
│ - Moved                 │
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ Event Store / Writers    │
└────────────┬─────────────┘
             ▼
 JSONL / snapshots / video
```

Evaluation chạy song song/offline:

```text
Ground Truth + Predictions
           │
           ▼
     Event Matcher
           │
           ▼
 TP / FP / FN / Precision / Recall / F1
```

---

## 3. Layer boundaries

### 3.1 Domain layer

Chứa pure Python entities và business concepts:

- `BBox`
- `Point`
- `FrameContext`
- `Detection`
- `Observation`
- `MemoryObject`
- `Event`
- `ObjectState`
- `EventType`

Không import:

- torch;
- ultralytics;
- cv2;
- model-specific result types.

### 3.2 Application layer

Điều phối use cases:

- calibration;
- process frame;
- update memory;
- update event engine;
- run benchmark.

### 3.3 Infrastructure layer

Adapter tới external frameworks:

- YOLO detector;
- ByteTrack / BoT-SORT;
- DINO encoder;
- OpenCV video IO;
- filesystem writers.

---

## 4. Core data contracts

### FrameContext

```python
@dataclass
class FrameContext:
    video_id: str
    frame_index: int
    timestamp_sec: float
    image: np.ndarray
    roi: "Polygon"
```

### Detection

```python
@dataclass
class Detection:
    bbox: "BBox"
    confidence: float
    class_id: int
    class_name: str
    mask: np.ndarray | None = None
```

### Observation

```python
@dataclass
class Observation:
    frame_index: int
    timestamp_sec: float
    bbox: "BBox"
    centroid: "Point"
    detector_class: str
    detector_confidence: float
    tracker_id: int | None
    embedding: np.ndarray | None
    occluded: bool = False
```

### MemoryObject

```python
@dataclass
class MemoryObject:
    object_id: UUID
    baseline_bbox: "BBox | None"
    baseline_embedding: np.ndarray | None
    last_bbox: "BBox"
    last_embedding: np.ndarray | None
    first_seen_sec: float
    last_seen_sec: float
    state: "ObjectState"
```

Persistent identity nằm ở `object_id`, không nằm ở `tracker_id`.

---

## 5. Component responsibilities

### FrameSource

Trách nhiệm:

- open/read/seek/close;
- timestamp mapping;
- source metadata.

Không chịu trách nhiệm detect/track.

### SceneBaseline / CalibrationService

Trách nhiệm:

- kiểm tra scene stability;
- thu calibration frames;
- xây reference frame;
- detect baseline objects;
- encode baseline appearance;
- persist baseline.

### SceneMonitor

Theo dõi:

- lighting/global change;
- camera shift;
- frozen frame;
- blur nếu cần;
- global change ratio.

Output được dùng như guard condition cho EventEngine.

### Detector

Interface:

```python
class Detector(Protocol):
    def detect(self, frame: np.ndarray, roi: "Polygon") -> list[Detection]: ...
```

MVP adapter: YOLO pretrained.

### Tracker

Chỉ làm short-term temporal linking.

Interface:

```python
class Tracker(Protocol):
    def update(self, detections: list[Detection], frame: np.ndarray) -> list["Track"]: ...
```

MVP: ByteTrack. BoT-SORT dùng làm comparison.

### FeatureEncoder

Sinh appearance embedding cho observation/crop.

MVP: DINOv2 frozen.

Encoder có thể semantic-refresh thấp hơn processing FPS để tiết kiệm compute.

### AssociationEngine

Input:

- current observations;
- memory objects.

Output:

- matched pairs;
- unmatched observations;
- unmatched memory objects.

Cost function khởi đầu:

```text
cost =
    w_appearance * appearance_cost
  + w_spatial    * spatial_cost
  + w_size       * size_cost
  + w_class      * class_cost
```

Sau gating, dùng Hungarian matching hoặc equivalent assignment.

### ObjectMemory

Source of truth cho persistent identity.

Quản lý:

- baseline objects;
- currently present objects;
- temporarily missing objects;
- new objects;
- observation history;
- appearance history;
- last-known position.

### EventEngine

Pure state-based event logic.

Không gọi model trực tiếp.

Input:

- ObjectMemory state;
- SceneStatus;
- timestamp.

Output:

- event create/update/close actions.

### EventStore

Quản lý lifecycle và deduplication.

### Writers

- JSONL writer;
- snapshot writer;
- video writer;
- metric/report writer.

---

## 6. Runtime sequence

```text
frame
  │
  ├─► SceneMonitor.update()
  │
  ├─► Detector.detect()
  │
  ├─► Tracker.update()
  │
  ├─► ObservationBuilder.build()
  │
  ├─► FeatureEncoder.encode(selected observations)
  │
  ├─► AssociationEngine.match()
  │
  ├─► ObjectMemory.update()
  │
  ├─► EventEngine.update()
  │
  ├─► EventStore.apply()
  │
  └─► Writers.write()
```

Pseudo-code:

```python
def process_frame(frame: FrameContext) -> None:
    scene = scene_monitor.update(frame)
    detections = detector.detect(frame.image, frame.roi)
    tracks = tracker.update(detections, frame.image)
    observations = observation_builder.build(frame, detections, tracks)
    feature_encoder.enrich(observations)

    associations = associator.match(memory.objects, observations)
    memory.update(observations, associations, scene)

    event_actions = event_engine.update(memory, scene, frame.timestamp_sec)
    event_store.apply(event_actions)
    writers.write(frame, memory, event_store)
```

---

## 7. Error isolation strategy

Mỗi stage phải debug độc lập:

1. Detector error — object không được detect.
2. Tracker error — short-term identity đứt.
3. Encoder error — appearance không phân biệt được.
4. Association error — match sai memory object.
5. Memory/state error — state transition sai.
6. Event error — event tạo quá sớm/quá muộn/trùng.
7. Evaluator error — matching prediction↔GT sai.

Log phải đủ để phân biệt 7 nhóm này.

---

## 8. Performance strategy

MVP không cần chạy mọi model trên mọi frame.

Khuyến nghị:

- detector: processing FPS target;
- encoder: chỉ cho selected/candidate observations;
- semantic refresh thấp hơn detector FPS;
- scene monitor dùng phép tính nhẹ;
- snapshot/video writer async nội bộ nếu sau này cần, nhưng core logic vẫn synchronous/deterministic cho test.

---

## 9. Extensibility

Các extension dự kiến:

- `YoloDetector` → `RTDETRDetector`;
- `ByteTrackAdapter` → `BoTSORTAdapter`;
- `DINOv2Encoder` → `DINOv3Encoder`;
- bbox detection → segmentation;
- optional SAM-based candidate refinement;
- RTSP source;
- multi-camera orchestration.

Không extension nào được thay đổi event semantics nếu product requirements không đổi.
