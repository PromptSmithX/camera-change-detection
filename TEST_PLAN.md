# Test Plan — Change Detection System V2

## 1. Test strategy

Có 4 tầng test:

1. Unit tests — pure logic, không model/GPU.
2. Integration tests — module interaction.
3. Scenario tests — mô phỏng use case/event.
4. Dataset benchmark — validation/test thực tế.

---

## 2. Unit tests

### Domain / geometry

- bbox clipping;
- IoU;
- centroid;
- normalized distance;
- ROI containment.

### Association

- appearance similarity;
- gating;
- cost matrix;
- matched/unmatched behavior;
- ambiguity handling.

### ObjectMemory

- same identity across tracker ID switch;
- short dropout giữ same object_id;
- stale object transition;
- new object creation.

### Event FSM

- forgotten candidate → confirmed;
- forgotten short-lived → no event;
- moved candidate → confirmed;
- object returns old position → no moved;
- duplicate prevention.

---

## 3. Integration tests

- video source → pipeline frame context;
- detector adapter → domain Detection;
- tracker adapter → observation;
- encoder → embedding enrichment;
- association → memory update;
- event engine → EventStore;
- EventStore → JSONL/snapshot/video writers.

---

## 4. Required scenario tests

### Input / source

1. App/pipeline khởi động.
2. Mở video hợp lệ.
3. Mở webcam hợp lệ.
4. Nguồn video không tồn tại → error rõ ràng.
5. Seek/reset → tracker/event counters reset đúng.

### ROI / calibration

6. ROI hợp lệ.
7. ROI vượt biên được clip.
8. Calibration scene ổn định → baseline tạo được.
9. Calibration scene đang chuyển động → reject.

### FORGOTTEN_OBJECT

10. Object mới xuất hiện và giữ yên đủ lâu → event.
11. Object mới chỉ xuất hiện ngắn → no event.
12. Event dài nhiều frame → chỉ một event lifecycle.

### MOVED_OBJECT

13. Baseline object biến mất và cùng identity xuất hiện vị trí mới → event.
14. Baseline object tạm mất rồi trở lại vị trí cũ → no event.
15. New unrelated object + missing baseline object → không được tự động kết luận moved nếu identity score thấp.

### Robustness

16. Người đi qua che object → no false event trong grace window.
17. Tracker short dropout → không close/reopen event.
18. Tracker đổi ID → tái liên kết persistent object.
19. Global lighting change → không tạo burst candidates.
20. Nhiễu nhỏ sát biên ROI → không tạo invalid event.

### Output

21. Không có event → output events rỗng.
22. JSONL chỉ chứa public fields.
23. BEFORE/AFTER snapshots đúng event.
24. Annotated video hoàn tất và mở lại được.

### Dataset / benchmark

25. Dataset normalize/validate thành công.
26. Validation chỉ dùng validation samples.
27. Locked test không bị tuning tiếp.
28. Metrics được sinh đầy đủ.

### Performance / stability

29. Ghi processing FPS.
30. Chạy video dài không leak memory đáng kể.
31. Reset runtime và recalibration không giữ stale state.

---

## 5. Example deterministic tests

```python
def test_short_dropout_does_not_create_new_event():
    memory = ObjectMemory()

    feed_object(memory, object_id="obj1", tracker_id=1)
    feed_missing(memory, seconds=0.5)
    feed_object(memory, object_id="obj1", tracker_id=21)

    assert active_events() == []
```

```python
def test_moved_object_requires_identity_match():
    baseline("obj1", position="A", embedding="e1")
    remove("obj1")
    add_new(position="B", embedding="unrelated")

    run_for(seconds=3)

    assert no_event("MOVED_OBJECT")
```

---

## 6. Severity

- P0: không chạy, crash, mất dữ liệu, không thể test.
- P1: sai chức năng chính, bỏ sót/phát sai nghiêm trọng.
- P2: chức năng phụ sai/chưa ổn định.
- P3: UI/text/cải tiến nhỏ không ảnh hưởng nghiệp vụ.

---

## 7. Definition of done for a feature

Feature chỉ được done khi:

- có spec;
- có unit/scenario tests phù hợp;
- có log/debug visibility;
- không phá benchmark regression ngoài mức đã chấp nhận;
- output reproducible trên cùng config/dataset version.
