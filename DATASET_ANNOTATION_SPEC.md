# Dataset & Annotation Specification

## 1. Purpose

Chuẩn hóa dataset để:

- evaluator reproducible;
- tránh data leakage;
- debug được event-level;
- support hard-negative analysis;
- support future fine-tuning khi cần.

Dataset hiện có khoảng 40 video, vì vậy split và annotation quality quan trọng hơn việc tăng model size.

---

## 2. Directory structure

```text
dataset/
├── manifest.json
├── videos/
│   ├── vid_001.mp4
│   └── ...
├── annotations/
│   ├── vid_001.json
│   └── ...
├── roi/
│   ├── cam_001.json
│   └── ...
└── baselines/
    └── optional cached calibration artifacts
```

---

## 3. Manifest schema

```json
{
  "dataset_version": "1.0",
  "samples": [
    {
      "video_id": "vid_001",
      "camera_id": "cam_01",
      "path": "videos/vid_001.mp4",
      "annotation_path": "annotations/vid_001.json",
      "roi_path": "roi/cam_01.json",
      "split": "validation",
      "scenario_tags": ["occlusion", "moved_object"]
    }
  ]
}
```

---

## 4. Annotation schema

```json
{
  "video_id": "vid_001",
  "camera_id": "cam_01",
  "fps": 25.0,
  "events": [
    {
      "event_id": "event_01",
      "type": "MOVED_OBJECT",
      "object_id": "obj_01",
      "start_time_sec": 12.1,
      "confirmation_time_sec": 15.0,
      "end_time_sec": null,
      "baseline_bbox": [100, 150, 170, 260],
      "new_bbox": [420, 155, 490, 265],
      "movement_outcome": "relocated",
      "occlusion_intervals": [[12.5, 13.1]],
      "notes": "optional reviewer note"
    }
  ]
}
```

---

## 5. Required annotation fields

### Per video

- `video_id`
- `camera_id`
- ROI reference
- duration/FPS metadata
- scenario tags

### Per event

- `event_id`
- `type`
- `object_id`
- `start_time_sec`
- `confirmation_time_sec` nếu định nghĩa được
- `end_time_sec` nếu event kết thúc
- relevant bbox/mask before/after

### MOVED_OBJECT specific

- `baseline_bbox`
- `movement_outcome`: `relocated` hoặc `left_scene`
- `new_bbox` bắt buộc khi `movement_outcome = relocated`; phải là `null` khi `movement_outcome = left_scene`
- same `object_id`

### Optional but high-value

- occlusion intervals;
- lighting-change intervals;
- difficult/ambiguous flag;
- object class label;
- reviewer notes.

---

## 6. Split policy

### Rule 1 — Never split random frames

Các frame cùng video quá tương đồng và sẽ tạo leakage.

### Rule 2 — Split theo video

Tất cả frame/event của một video chỉ thuộc một split.

### Rule 3 — Nếu có nhiều camera, cân nhắc group theo camera

Nếu mục tiêu là generalize sang camera unseen, camera không được xuất hiện ở cả train/dev và test.

### Starting split for ~40 videos

Đề xuất ban đầu:

- 8 video: locked test;
- 32 video: development pool.

Development pool có thể chạy 4-fold grouped CV hoặc train/validation theo video.

Con số này là starting point, cần đảm bảo mỗi split có đủ FORGOTTEN/MOVED và hard cases.

---

## 7. Scenario coverage

Dataset nên có tags cho:

- forgotten object;
- moved object;
- no-event negative;
- person occlusion;
- tracker dropout;
- ID switch;
- global lighting change;
- ROI edge noise;
- short-lived new object;
- long event;
- long video;
- small object nếu nằm trong target scope.

---

## 8. Hard-negative policy

Hard negatives phải được giữ lại và annotate/tag rõ, ví dụ:

- người đi qua;
- bóng đổ;
- ánh sáng thay đổi;
- object chỉ xuất hiện ngắn;
- reflection;
- object ở sát biên ROI;
- detector hallucination;
- bbox jitter.

Không xóa hard negatives chỉ để metric đẹp hơn.

---

## 9. Dataset validation

`tools/validate_dataset.py` phải kiểm tra:

- video tồn tại;
- annotation parse được;
- event type hợp lệ;
- timestamp nằm trong duration;
- bbox hợp lệ;
- ROI hợp lệ;
- duplicate ID;
- split hợp lệ;
- test split không xuất hiện trong tuning manifest.

---

## 10. Versioning

Mỗi thay đổi annotation phải bump dataset version.

Không overwrite annotation cũ mà không có trace.

Khuyến nghị:

```text
dataset_version: 1.0.0
annotation_schema_version: 1.0
```

---

## 11. Canonical implementation profile

The first implementation stores the migrated benchmark in
`data/benchmark/v1/`. The source files under `data/processed/` remain
read-only inputs to migration.

- Manifest sample keys are exactly `video_id`, `camera_id`, `path`,
  `annotation_path`, `roi_path`, `split`, and `scenario_tags`.
- Event timing uses `start_time_sec`, `confirmation_time_sec`, and
  `end_time_sec` as the public contract. `start_frame`, `confirmation_frame`,
  and `end_frame` are traceability metadata and must agree with FPS when they
  are present.
- `FORGOTTEN_OBJECT` requires the confirmation `bbox`. `MOVED_OBJECT`
  requires `baseline_bbox`, `movement_outcome`, and one persistent `object_id`
  after review. `new_bbox` is required only for the `relocated` outcome; it is
  null for `left_scene`.
- `quality_status` is one of `verified`, `provisional`, or `excluded`.
  Draft migration may contain provisional records; official benchmark and
  locked test artifacts may contain verified records only.
- Annotation drafts are written to `data/benchmark/v1/reviews/` by the local
  review tool. The tool does not load model predictions, so ground truth is
  not tuned against a pipeline output.
- `tools/lock_test.py` creates a separate locked manifest and hash only after
  official validation has no errors or warnings and all test records are
  verified. Any later ground-truth edit requires a dataset-version bump and a
  new lock.
