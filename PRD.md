# PRD — Change Detection System V2

## 1. Document status

- Status: Draft for implementation
- Product: Fixed Camera Change Detection System V2
- Primary use case: phát hiện vật thể bị bỏ quên và vật thể bị di chuyển trong vùng ROI của camera cố định
- Target runtime: Python 3.13, GPU recommended, CPU fallback cho functional testing
- Target processing profile ban đầu: 960×540, khoảng 5 FPS, semantic refresh khoảng 2.5 FPS

---

## 2. Problem statement

Hệ thống cần phát hiện thay đổi có ý nghĩa trên camera cố định, thay vì chỉ phát hiện object frame-by-frame.

Hai bài toán chính:

1. `FORGOTTEN_OBJECT`: một vật thể mới xuất hiện trong ROI và được để lại đủ lâu.
2. `MOVED_OBJECT`: một vật thể vốn thuộc baseline bị di chuyển khỏi vị trí ban đầu.

Khó khăn chính không chỉ nằm ở detector mà còn ở:

- object identity xuyên qua tracker dropout / ID switch;
- occlusion bởi người;
- thay đổi ánh sáng toàn cảnh;
- temporal confirmation;
- phân biệt object mới với object đã di chuyển;
- tránh duplicate event;
- đánh giá chính xác ở event-level.

---

## 3. Goals

### G1 — Functional correctness

Hệ thống phải:

- mở video file và webcam;
- cho phép chọn ROI;
- hiệu chuẩn scene và tạo baseline;
- phát hiện `FORGOTTEN_OBJECT`;
- phát hiện `MOVED_OBJECT`;
- chịu được short tracker dropout;
- tái liên kết object khi tracker đổi ID;
- tránh FP khi người che object trong khoảng cho phép;
- hạn chế FP do thay đổi ánh sáng toàn cảnh;
- tạo JSONL, snapshot BEFORE/AFTER và video annotated.

### G2 — Measurability

Mọi phiên bản pipeline phải benchmark được trên validation/test với:

- Precision
- Recall
- F1
- TP
- FP
- FN
- metric riêng cho FORGOTTEN_OBJECT
- metric riêng cho MOVED_OBJECT

### G3 — Maintainability

Có thể thay:

- YOLO → detector khác
- ByteTrack → BoT-SORT / tracker khác
- DINOv2 → DINOv3 / encoder khác

mà không phải viết lại ObjectMemory hoặc EventEngine.

### G4 — Reproducibility

Mỗi run phải lưu:

- config;
- model/version info;
- events;
- metrics;
- snapshots;
- annotated video;
- logs.

---

## 4. Non-goals for MVP

Không nằm trong phạm vi MVP:

- camera PTZ;
- camera rung mạnh;
- crowd density cao;
- multi-camera identity;
- vật thể cực nhỏ;
- occlusion kéo dài không giới hạn;
- thay đổi day/night cực mạnh;
- pentest chuyên sâu;
- inference nhiều camera đồng thời ở production scale.

---

## 5. Users / stakeholders

### Developer / CV Engineer

Cần:

- chạy experiment nhanh;
- debug event theo từng frame/time;
- thay model không sửa business logic;
- benchmark reproducible.

### QA / Tester

Cần:

- scenario rõ ràng;
- expected event/output rõ ràng;
- logs và snapshot để đối chiếu.

### Reviewer / Business owner

Cần:

- KPI định lượng;
- output dễ kiểm tra;
- không kết luận “đạt” nếu chưa có bằng chứng benchmark.

---

## 6. Core user flows

### Flow A — Calibration

1. Mở nguồn video/camera.
2. Chọn ROI.
3. System kiểm tra scene ổn định.
4. Thu N frame calibration.
5. Xây reference image / scene profile.
6. Detect baseline objects.
7. Encode baseline embeddings.
8. Persist `SceneBaseline`.

Nếu scene đang chuyển động mạnh, calibration phải fail hoặc yêu cầu thử lại.

### Flow B — Runtime detection

1. Đọc frame.
2. Preprocess + ROI.
3. Scene monitor đánh giá lighting/global anomalies.
4. Detector tạo detections.
5. Tracker tạo short-term tracks.
6. Encoder tạo embeddings cho observations cần thiết.
7. AssociationEngine match observations ↔ memory objects.
8. ObjectMemory update state.
9. EventEngine update FSM.
10. EventStore create/update/close event.
11. Writer xuất JSONL/snapshot/video.

### Flow C — Evaluation

1. Load locked config/model.
2. Chạy prediction trên validation hoặc test split.
3. Match predicted events với ground truth events.
4. Tính metric tổng và theo event type.
5. Sinh error analysis theo nguyên nhân.

---

## 7. Functional requirements

### FR-001 Source input

Support:

- `.mp4` / video file;
- webcam;
- thiết kế interface sẵn để mở rộng RTSP.

### FR-002 ROI

- ROI hợp lệ được dùng trong toàn bộ pipeline.
- ROI vượt biên phải được clip về source dimensions.

### FR-003 Calibration

- Chỉ tạo baseline khi scene đủ ổn định.
- Baseline phải lưu reference scene và baseline objects.

### FR-004 Forgotten object

System phát event khi object mới:

- không match baseline object hợp lệ;
- nằm trong ROI;
- tồn tại ổn định >= configured duration;
- không bị loại do occlusion/global scene anomaly.

### FR-005 Moved object

System phát event khi baseline object:

- không còn tại vị trí baseline;
- có candidate ở vị trí mới;
- candidate có identity similarity đủ cao;
- displacement đủ lớn;
- trạng thái ổn định đủ lâu.

### FR-006 Occlusion

Short occlusion bởi person/object khác không được tạo event sai.

### FR-007 Tracker dropout / ID switch

Tracker ID không được dùng làm persistent object identity.

### FR-008 Global lighting change

Thay đổi ánh sáng toàn cảnh không được sinh hàng loạt candidates/event giả.

### FR-009 Event deduplication

Một event kéo dài nhiều frame chỉ được tạo một event lifecycle duy nhất.

### FR-010 Output

Mỗi run phải hỗ trợ:

- JSONL public events;
- snapshot before/after;
- annotated video;
- runtime logs.

---

## 8. Non-functional requirements

### NFR-001 Modularity

Core domain không import framework CV/ML cụ thể.

### NFR-002 Deterministic testing

Event/state logic phải unit test được không cần GPU/model.

### NFR-003 Performance

Mục tiêu runtime MVP:

- input profile: 960×540;
- target processing: khoảng 5 FPS;
- semantic encoder có thể chạy ở lower refresh rate;
- không leak memory khi chạy video dài.

### NFR-004 Configuration

Threshold, timeout, weight phải nằm trong YAML/config, không hardcode.

### NFR-005 Observability

Logs phải cho phép trace:

- frame/time;
- observation;
- memory object;
- association score;
- state transition;
- event transition.

---

## 9. Acceptance criteria for V2 MVP

V2 MVP được xem là implementation-complete khi:

- toàn bộ pipeline chạy end-to-end trên video;
- evaluator chạy được trên dataset chuẩn hóa;
- FORGOTTEN_OBJECT và MOVED_OBJECT đều có automated scenario tests;
- tracker ID switch không tự tạo object mới nếu appearance/spatial matching xác nhận cùng identity;
- short occlusion/dropout không tạo duplicate event;
- global lighting change không tạo burst event;
- output JSONL/snapshot/video hợp lệ;
- locked validation benchmark được sinh tự động;
- test set không được dùng để tuning.

KPI nghiệm thu cuối cùng phải được chốt riêng; baseline hiện tại chưa đạt mục tiêu 70% đã nêu trong kế hoạch test cũ.
