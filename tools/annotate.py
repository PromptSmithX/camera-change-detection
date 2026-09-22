"""Run the local single-reviewer annotation application.

The application writes review drafts to ``data/benchmark/v1/reviews`` and does
not overwrite the migrated annotation or the original data/processed files.

Install the review extras first:
    python -m pip install -e ".[review]"

Run:
    python tools/annotate.py --manifest data/benchmark/v1/manifest.json
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, Response
    import uvicorn
except ImportError as exc:  # pragma: no cover - helpful CLI error
    raise SystemExit("Annotation tool requires the review extras: python -m pip install -e .[review]") from exc

from change_detection.dataset.io import read_json, resolve_repo_path, write_json


def _natural_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _bbox_text(value: Any) -> str:
    if not isinstance(value, list) or len(value) != 4:
        return ""
    return ",".join(str(int(item)) for item in value)


class ReviewStore:
    def __init__(self, root: Path, manifest_path: Path, *, allow_test_edit: bool = False) -> None:
        self.root = root.resolve()
        self.manifest_path = manifest_path.resolve()
        self.manifest = read_json(self.manifest_path)
        self.allow_test_edit = allow_test_edit
        self.samples = {str(item["video_id"]): item for item in self.manifest.get("samples", [])}
        self.annotation_cache: dict[str, dict[str, Any]] = {}
        self.review_dir = self.root / "data" / "benchmark" / "v1" / "reviews"

    def sample(self, video_id: str) -> dict[str, Any]:
        if video_id not in self.samples:
            raise KeyError(video_id)
        return self.samples[video_id]

    def annotation(self, video_id: str) -> dict[str, Any]:
        if video_id not in self.annotation_cache:
            sample = self.sample(video_id)
            path = resolve_repo_path(self.root, str(sample["annotation_path"]))
            self.annotation_cache[video_id] = read_json(path)
        value = json.loads(json.dumps(self.annotation_cache[video_id]))
        review_path = self.review_dir / f"{video_id}.json"
        if review_path.exists():
            value = read_json(review_path)
        return value

    def save_review(self, video_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        sample = self.sample(video_id)
        if sample.get("split") == "test" and self.manifest.get("test_locked") and not self.allow_test_edit:
            raise PermissionError("The test split is locked; bump dataset version before editing it.")
        if str(payload.get("video_id")) != video_id:
            raise ValueError("Payload video_id does not match route")
        payload["reviewer"] = payload.get("reviewer") or "local_reviewer"
        self.review_dir.mkdir(parents=True, exist_ok=True)
        review_path = self.review_dir / f"{video_id}.json"
        write_json(review_path, payload)
        return payload

    def frame_path(self, video_id: str, frame_index: int) -> Path | None:
        annotation = self.annotation(video_id)
        sample = self.sample(video_id)
        if annotation.get("media_type") != "image_sequence":
            return None
        directory = resolve_repo_path(self.root, str(sample["path"]))
        pattern = annotation.get("frame_pattern") or "*"
        files = sorted(directory.glob(pattern), key=_natural_key)
        if frame_index < 0 or frame_index >= len(files):
            raise IndexError(frame_index)
        return files[frame_index]

    def frame_bytes(self, video_id: str, frame_index: int) -> bytes:
        annotation = self.annotation(video_id)
        sample = self.sample(video_id)
        if annotation.get("media_type") == "image_sequence":
            image_path = self.frame_path(video_id, frame_index)
            if image_path is None:
                raise FileNotFoundError("No image frame")
            try:
                from PIL import Image
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Pillow is required for the review tool") from exc
            image = Image.open(image_path).convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=92)
            return buffer.getvalue()
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("opencv-python is required to preview video files") from exc
        video_path = resolve_repo_path(self.root, str(sample["path"]))
        capture = cv2.VideoCapture(str(video_path))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise IndexError(frame_index)
            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                raise RuntimeError("Could not encode video frame")
            return encoded.tobytes()
        finally:
            capture.release()


HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Change Detection V2 — Annotation Review</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 0; color: #202124; }
    header { padding: 12px 18px; background: #1f2937; color: white; }
    main { display: grid; grid-template-columns: minmax(520px, 1fr) 390px; gap: 16px; padding: 16px; }
    .panel { border: 1px solid #d1d5db; border-radius: 8px; padding: 12px; }
    label { display: block; margin-top: 8px; font-size: 12px; color: #4b5563; }
    input, select, textarea, button { box-sizing: border-box; width: 100%; padding: 7px; margin-top: 3px; }
    button { cursor: pointer; }
    .row { display: flex; gap: 8px; }
    .row > * { flex: 1; }
    #stage { position: relative; display: inline-block; max-width: 100%; background: #111827; }
    #frame { display: block; max-width: 100%; height: auto; }
    #overlay { position: absolute; inset: 0; cursor: crosshair; }
    #status { padding: 8px; background: #f3f4f6; margin-top: 8px; min-height: 20px; }
    .hint { font-size: 12px; color: #6b7280; }
  </style>
</head>
<body>
<header><strong>Change Detection V2 — Annotation Review</strong><div class="hint">No predictions are loaded by this tool. Reviews are saved separately from the source annotations.</div></header>
<main>
  <section class="panel">
    <div class="row"><select id="sample"></select><input id="frame" type="number" min="0" value="0" /></div>
    <div class="row"><button id="prev">Previous frame</button><button id="next">Next frame</button><button id="load">Load frame</button></div>
    <div id="stage"><img id="frameImage" alt="frame" /><canvas id="overlay"></canvas></div>
    <div class="hint">Choose a bbox field, then drag on the image. Coordinates are stored as integer half-open pixel boxes.</div>
    <div id="status"></div>
  </section>
  <section class="panel">
    <label>Sample review</label>
    <div class="row"><div><label>Status</label><select id="sampleQuality"><option>provisional</option><option>verified</option><option>excluded</option></select></div><div><label>Reviewer</label><input id="reviewer" value="local_reviewer" /></div></div>
    <label>Sample review notes</label><textarea id="reviewNotes" rows="2"></textarea>
    <label>Unresolved provenance warnings (one per line)</label><textarea id="warnings" rows="3"></textarea>
    <label><input id="warningsResolved" type="checkbox" style="width:auto" /> I reviewed and resolved the listed warnings</label>
    <label>Scenario tags (comma-separated)</label><input id="scenarioTags" />
    <div class="row"><div><label>Reference start frame</label><input id="referenceStart" type="number" min="0" /></div><div><label>Reference end frame</label><input id="referenceEnd" type="number" min="0" /></div></div>
    <label>Event</label><select id="event"></select>
    <div class="row"><div><label>Type</label><select id="type"><option>FORGOTTEN_OBJECT</option><option>MOVED_OBJECT</option></select></div><div><label>Quality</label><select id="quality"><option>provisional</option><option>verified</option><option>excluded</option></select></div></div>
    <label>Movement outcome (MOVED_OBJECT only)</label><select id="movementOutcome"><option value="">Select outcome</option><option value="relocated">relocated — stable new location</option><option value="left_scene">left_scene — carried/driven out of view</option></select>
    <div class="row"><div><label>Object ID</label><input id="objectId" /></div><div><label>Object class</label><input id="objectClass" /></div></div>
    <div class="row"><div><label>Start time (sec)</label><input id="startTime" type="number" step="0.001" /></div><div><label>Confirmation time (sec)</label><input id="confirmationTime" type="number" step="0.001" /></div></div>
    <div class="row"><div><label>End time (sec)</label><input id="endTime" type="number" step="0.001" /></div><div><label>Frame field</label><select id="boxField"><option value="bbox">confirmation bbox</option><option value="baseline_bbox">baseline bbox</option><option value="new_bbox">new bbox</option><option value="roi">sample ROI</option></select></div></div>
    <div class="row"><div><label>Start frame</label><input id="startFrame" type="number" min="0" /></div><div><label>Confirmation frame</label><input id="confirmationFrame" type="number" min="0" /></div><div><label>End frame</label><input id="endFrame" type="number" min="0" /></div></div>
    <div class="row"><button id="markStart">Mark start at current frame</button><button id="markConfirmation">Mark confirmation</button><button id="markEnd">Mark end</button></div>
    <label>BBox [x1,y1,x2,y2]</label><input id="bbox" placeholder="e.g. 100,120,180,220" />
    <div class="row"><label><input id="difficult" type="checkbox" style="width:auto" /> Difficult</label><label><input id="ambiguous" type="checkbox" style="width:auto" /> Ambiguous</label></div>
    <label>Notes</label><textarea id="notes" rows="4"></textarea>
    <button id="save">Save review draft</button>
    <button id="newEvent">Add event</button>
  </section>
</main>
<script>
let samples = [], current = null, selectedEvent = -1, drag = null;
const $ = id => document.getElementById(id);
function setStatus(text, error=false) { $('status').textContent = text; $('status').style.color = error ? '#b91c1c' : '#166534'; }
async function api(url, options) { const r = await fetch(url, options); if (!r.ok) throw new Error(await r.text()); return r.json(); }
function parseBox(text) { const v = text.split(',').map(x => Number(x.trim())); return v.length === 4 && v.every(Number.isFinite) ? v.map(Math.round) : null; }
function boxText(v) { return Array.isArray(v) ? v.join(',') : ''; }
function fillSample() { const warnings = Array.isArray(current.warnings) ? current.warnings : []; const reference = current.reference && Array.isArray(current.reference.frame_range) ? current.reference.frame_range : []; $('sampleQuality').value = current.quality_status || 'provisional'; $('reviewer').value = current.reviewer || 'local_reviewer'; $('reviewNotes').value = current.review_notes || ''; $('warnings').value = warnings.join('\n'); $('warningsResolved').checked = warnings.length === 0; $('scenarioTags').value = Array.isArray(current.scenario_tags) ? current.scenario_tags.join(', ') : ''; $('referenceStart').value = reference[0] ?? ''; $('referenceEnd').value = reference[1] ?? ''; }
function renderEvents() {
  fillSample();
  const select = $('event'); select.innerHTML = '';
  (current.events || []).forEach((e, i) => { const o = document.createElement('option'); o.value = i; o.textContent = `${i}: ${e.type} ${e.event_id}`; select.appendChild(o); });
  if ((current.events || []).length) { selectedEvent = Math.max(0, Math.min(selectedEvent, current.events.length - 1)); select.value = selectedEvent; fillEvent(); } else { $('bbox').value = boxText((current.roi || {}).bbox); drawBoxes(); }
}
function fillEvent() {
  const e = current.events[selectedEvent]; if (!e) return;
  $('type').value = e.type; $('quality').value = e.quality_status || current.quality_status || 'provisional'; $('objectId').value = e.object_id || '';
  $('objectClass').value = e.object_class || ''; $('movementOutcome').value = e.movement_outcome || ''; $('difficult').checked = Boolean(e.difficult); $('ambiguous').checked = Boolean(e.ambiguous);
  $('startTime').value = e.start_time_sec ?? ''; $('confirmationTime').value = e.confirmation_time_sec ?? ''; $('endTime').value = e.end_time_sec ?? '';
  $('startFrame').value = e.start_frame ?? ''; $('confirmationFrame').value = e.confirmation_frame ?? ''; $('endFrame').value = e.end_frame ?? '';
  $('notes').value = e.notes || ''; const field = $('boxField').value; $('bbox').value = boxText(field === 'roi' ? (current.roi || {}).bbox : e[field]); drawBoxes();
}
function applyEvent() {
  current.quality_status = $('sampleQuality').value; current.reviewer = $('reviewer').value.trim() || 'local_reviewer'; current.review_notes = $('reviewNotes').value;
  current.warnings = $('warningsResolved').checked ? [] : $('warnings').value.split('\n').map(x => x.trim()).filter(Boolean);
  current.scenario_tags = $('scenarioTags').value.split(',').map(x => x.trim()).filter(Boolean);
  const referenceStart = $('referenceStart').value === '' ? null : Number($('referenceStart').value); const referenceEnd = $('referenceEnd').value === '' ? null : Number($('referenceEnd').value); current.reference = {frame_range: referenceStart === null || referenceEnd === null ? null : [referenceStart, referenceEnd]};
  const field = $('boxField').value; if (field === 'roi') { current.roi = current.roi || {}; current.roi.bbox = parseBox($('bbox').value); }
  const e = current.events[selectedEvent]; if (!e || field === 'roi') return;
  e.type = $('type').value; e.quality_status = $('quality').value; e.object_id = $('objectId').value.trim();
  e.movement_outcome = e.type === 'MOVED_OBJECT' ? ($('movementOutcome').value || null) : null;
  e.object_class = $('objectClass').value.trim() || null; e.difficult = $('difficult').checked; e.ambiguous = $('ambiguous').checked;
  e.start_time_sec = Number($('startTime').value); e.confirmation_time_sec = $('confirmationTime').value === '' ? null : Number($('confirmationTime').value); e.end_time_sec = $('endTime').value === '' ? null : Number($('endTime').value);
  e.start_frame = $('startFrame').value === '' ? null : Number($('startFrame').value); e.confirmation_frame = $('confirmationFrame').value === '' ? null : Number($('confirmationFrame').value); e.end_frame = $('endFrame').value === '' ? null : Number($('endFrame').value);
  e[field] = parseBox($('bbox').value); e.notes = $('notes').value;
}
function drawBoxes() {
  const canvas = $('overlay'), img = $('frameImage'); if (!img.naturalWidth) return;
  canvas.width = img.clientWidth; canvas.height = img.clientHeight; const ctx = canvas.getContext('2d'); ctx.clearRect(0,0,canvas.width,canvas.height);
  const sx = canvas.width / img.naturalWidth, sy = canvas.height / img.naturalHeight;
  const colors = {bbox:'#22c55e', baseline_bbox:'#f59e0b', new_bbox:'#ef4444', roi:'#60a5fa'};
  const roi = current && current.roi ? current.roi.bbox : null; if (Array.isArray(roi)) { ctx.strokeStyle=colors.roi; ctx.lineWidth=3; ctx.strokeRect(roi[0]*sx,roi[1]*sy,(roi[2]-roi[0])*sx,(roi[3]-roi[1])*sy); ctx.fillStyle=colors.roi; ctx.fillText('roi', roi[0]*sx+3, roi[1]*sy+14); }
  const e = current && current.events ? current.events[selectedEvent] : null; if (!e) return;
  for (const key of ['bbox','baseline_bbox','new_bbox']) { const b=e[key]; if (!Array.isArray(b)) continue; ctx.strokeStyle=colors[key]; ctx.lineWidth=2; ctx.strokeRect(b[0]*sx,b[1]*sy,(b[2]-b[0])*sx,(b[3]-b[1])*sy); ctx.fillStyle=colors[key]; ctx.fillText(key, b[0]*sx+3, b[1]*sy+14); }
}
async function loadFrame() { if (!current) return; const n = Math.max(0, Number($('frame').value) || 0); $('frameImage').src = `/api/samples/${encodeURIComponent(current.video_id)}/frame?frame=${n}&t=${Date.now()}`; }
async function loadSample() { current = await api(`/api/samples/${encodeURIComponent($('sample').value)}`); selectedEvent=-1; renderEvents(); $('frame').max = Math.max(0,current.frame_count-1); await loadFrame(); setStatus(`${current.video_id}: ${current.quality_status}`); }
async function save() { applyEvent(); try { await api(`/api/samples/${encodeURIComponent(current.video_id)}/review`, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(current)}); setStatus('Saved review draft.'); } catch(e) { setStatus(e.message, true); } }
 $('sample').onchange=loadSample; $('event').onchange=()=>{selectedEvent=Number($('event').value);fillEvent();}; $('boxField').onchange=()=>{const field=$('boxField').value; const event=current.events[selectedEvent] || {}; $('bbox').value=boxText(field === 'roi' ? (current.roi || {}).bbox : event[field]);drawBoxes();}; $('load').onclick=loadFrame; $('prev').onclick=()=>{$('frame').value=Math.max(0,Number($('frame').value)-1);loadFrame();}; $('next').onclick=()=>{$('frame').value=Math.min(Number($('frame').max),Number($('frame').value)+1);loadFrame();}; $('save').onclick=save; $('frameImage').onload=drawBoxes;
 function markFrame(field) { if (selectedEvent < 0) return; $(`${field}Frame`).value = Number($('frame').value); applyEvent(); }
 $('markStart').onclick=()=>markFrame('start'); $('markConfirmation').onclick=()=>markFrame('confirmation'); $('markEnd').onclick=()=>markFrame('end');
 $('newEvent').onclick=()=>{ current.events.push({event_id:`draft_${Date.now()}`,type:'FORGOTTEN_OBJECT',object_id:'',start_time_sec:Number($('frame').value)/current.fps,confirmation_time_sec:null,end_time_sec:null,start_frame:Number($('frame').value),confirmation_frame:null,end_frame:null,bbox:null,baseline_bbox:null,new_bbox:null,movement_outcome:null,quality_status:'provisional',notes:''}); selectedEvent=current.events.length-1;renderEvents(); };
['mousedown','mousemove','mouseup'].forEach(name => $('overlay').addEventListener(name, ev => { if (!current || (selectedEvent < 0 && $('boxField').value !== 'roi')) return; const rect=$('overlay').getBoundingClientRect(); const x=Math.round((ev.clientX-rect.left)*$('frameImage').naturalWidth/$('overlay').width), y=Math.round((ev.clientY-rect.top)*$('frameImage').naturalHeight/$('overlay').height); if(name==='mousedown') drag={x,y}; if(name==='mousemove'&&drag){const b=[Math.min(drag.x,x),Math.min(drag.y,y),Math.max(drag.x,x),Math.max(drag.y,y)];$('bbox').value=b.join(',');drawBoxes();} if(name==='mouseup'&&drag){drag=null;applyEvent();drawBoxes();}}));
(async()=>{try{samples=await api('/api/samples');$('sample').innerHTML=samples.map(s=>`<option value="${s.video_id}">${s.video_id} [${s.split}]</option>`).join('');await loadSample();}catch(e){setStatus(e.message,true);}})();
</script>
</body>
</html>
"""


def create_app(store: ReviewStore) -> FastAPI:
    app = FastAPI(title="Change Detection V2 Annotation Review")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return HTML

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "samples": len(store.samples), "test_locked": bool(store.manifest.get("test_locked"))}

    @app.get("/api/samples")
    def samples() -> list[dict[str, Any]]:
        return [
            {
                "video_id": sample["video_id"],
                "split": sample["split"],
                "quality_status": sample.get("quality_status", "provisional"),
                "frame_count": sample.get("frame_count"),
            }
            for sample in store.samples.values()
        ]

    @app.get("/api/samples/{video_id}")
    def sample(video_id: str) -> dict[str, Any]:
        try:
            return store.annotation(video_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown sample") from exc

    @app.get("/api/samples/{video_id}/frame")
    def frame(video_id: str, frame: int = 0) -> Response:
        try:
            return Response(store.frame_bytes(video_id, frame), media_type="image/jpeg")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown sample") from exc
        except IndexError as exc:
            raise HTTPException(status_code=416, detail=f"Frame unavailable: {frame}") from exc
        except (FileNotFoundError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/samples/{video_id}/review")
    def review(video_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return store.save_review(video_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown sample") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/benchmark/v1/manifest.json")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--allow-test-edit", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    store = ReviewStore(root, manifest.resolve(), allow_test_edit=args.allow_test_edit)
    uvicorn.run(create_app(store), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
