"""Run the M4/M5 pipeline and correctness evaluation for one dataset split.

Example:
    python tools/run_validation.py --config configs/m45.example.json

Each sample is calibrated from its annotated reference-frame range and then
processed in a separate child process. Outputs are isolated under ``runs/``;
``--resume`` only reuses a run when its inputs and pipeline code still match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from change_detection.dataset.io import read_json, resolve_repo_path, sha256_file, write_json


def _verify_test_lock(manifest_path: Path, lock_path: Path) -> None:
    """Fail closed unless the selected test manifest matches its lock receipt."""

    manifest = read_json(manifest_path)
    lock = read_json(lock_path)
    if not bool(manifest.get("test_locked", False)):
        raise ValueError("Test split requires a manifest marked test_locked=true")
    if not bool(lock.get("locked", False)):
        raise ValueError("Test lock receipt is not marked locked=true")
    if str(lock.get("manifest_sha256", "")) != sha256_file(manifest_path):
        raise ValueError("Locked test manifest hash does not match its lock receipt")
    if lock.get("dataset_version") != manifest.get("dataset_version"):
        raise ValueError("Locked test dataset version does not match its lock receipt")
    if lock.get("test_sha256") != manifest.get("locked_test_sha256"):
        raise ValueError("Locked test annotation digest does not match its lock receipt")


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def _inside_root(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Path must be inside repository root: {path}") from exc
    return resolved


def _pipeline_digest(root: Path) -> str:
    files = set((root / "src" / "change_detection").rglob("*.py"))
    files.update(
        root / "tools" / name
        for name in (
            "calibrate.py",
            "run_change_detection.py",
            "benchmark_dataset.py",
            "run_validation.py",
        )
    )
    digest = hashlib.sha256()
    for path in sorted(files):
        if not path.is_file():
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _source_signature(root: Path, sample: Mapping[str, Any], annotation: Mapping[str, Any]) -> dict[str, Any]:
    source = resolve_repo_path(root, str(sample["path"]))
    signature: dict[str, Any] = {"path": str(sample["path"]), "exists": source.exists()}
    if not source.exists():
        return signature
    if source.is_file():
        stat = source.stat()
        signature.update({"kind": "file", "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        return signature

    pattern = str(annotation.get("frame_pattern") or sample.get("frame_pattern") or "*")
    entries = sorted(item for item in source.glob(pattern) if item.is_file())
    digest = hashlib.sha256()
    for item in entries:
        stat = item.stat()
        digest.update(item.name.encode("utf-8", errors="replace"))
        digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("ascii"))
    signature.update(
        {
            "kind": "image_sequence",
            "frame_pattern": pattern,
            "frame_count": len(entries),
            "listing_sha256": digest.hexdigest(),
        }
    )
    return signature


def _sample_fingerprint(
    root: Path,
    sample: Mapping[str, Any],
    annotation: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    template_sha256: str,
    pipeline_sha256: str,
) -> str:
    annotation_path = resolve_repo_path(root, str(sample["annotation_path"]))
    roi_path = resolve_repo_path(root, str(sample["roi_path"]))
    payload = {
        "sample": dict(sample),
        "manifest_dataset_version": manifest.get("dataset_version"),
        "annotation_sha256": sha256_file(annotation_path),
        "roi_sha256": sha256_file(roi_path),
        "source": _source_signature(root, sample, annotation),
        "template_sha256": template_sha256,
        "pipeline_sha256": pipeline_sha256,
    }
    return _canonical_digest(payload)


def _safe_sample_id(sample: Mapping[str, Any]) -> str:
    sample_id = str(sample.get("video_id", "")).strip()
    if not sample_id or Path(sample_id).name != sample_id or sample_id in {".", ".."}:
        raise ValueError(f"Invalid video_id in manifest: {sample_id!r}")
    return sample_id


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _build_sample_config(
    root: Path,
    run_root: Path,
    sample: Mapping[str, Any],
    annotation: Mapping[str, Any],
    template: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, Path]:
    sample_id = _safe_sample_id(sample)
    config = json.loads(json.dumps(template))
    media_type = str(annotation.get("media_type", sample.get("media_type", "video")))
    source_type = "image_sequence" if media_type == "image_sequence" else "video"
    fps_value = sample.get("fps")
    if fps_value is None:
        fps_value = annotation.get("fps")
    fps = float(fps_value)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid FPS for {sample_id}: {fps_value!r}")
    source: dict[str, Any] = {
        "type": source_type,
        "path": str(sample["path"]),
        "source_id": sample_id,
    }
    if source_type == "image_sequence":
        source["frame_pattern"] = str(
            annotation.get("frame_pattern") or sample.get("frame_pattern") or "*"
        )
        source["fps"] = fps

    calibration_template = dict(config.get("calibration", {}))
    default_sample_count = int(calibration_template.get("sample_count", 30))
    if default_sample_count < 2:
        raise ValueError("Base calibration.sample_count must be at least 2")
    reference_range = (annotation.get("reference") or {}).get("frame_range")
    if reference_range is not None:
        if not isinstance(reference_range, (list, tuple)) or len(reference_range) != 2:
            raise ValueError(f"Invalid reference.frame_range for {sample_id}: {reference_range!r}")
        first_frame, last_frame = (int(reference_range[0]), int(reference_range[1]))
        frame_count = int(sample.get("frame_count", annotation.get("frame_count", 0)))
        if first_frame < 0 or last_frame <= first_frame or (frame_count and last_frame >= frame_count):
            raise ValueError(f"Invalid reference.frame_range for {sample_id}: {reference_range!r}")
        sample_count = min(default_sample_count, last_frame - first_frame + 1)
        if sample_count < 2:
            raise ValueError(f"Reference range for {sample_id} must contain at least two frames")
        sample_interval = ((last_frame - first_frame) / fps) / (sample_count - 1)
        calibration_template.update(
            {
                "start_frame": first_frame,
                "sample_count": sample_count,
                "sample_interval_seconds": sample_interval,
            }
        )
    else:
        calibration_template.setdefault("start_frame", 0)

    baseline_dir = run_root / "baselines" / sample_id
    prediction_dir = run_root / "predictions" / sample_id
    baseline_path = baseline_dir / "baseline.json"
    config["source"] = source
    config["roi"] = {"type": "bbox", "path": str(sample["roi_path"])}
    config["baseline"] = {"path": _relative(root, baseline_path)}
    config["calibration"] = calibration_template
    runtime = dict(config.get("runtime", {}))
    runtime["start_frame"] = None
    config["runtime"] = runtime
    return config, baseline_dir, prediction_dir


def _redact_embeddings(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: ("[omitted]" if key == "embedding" else _redact_embeddings(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_embeddings(item) for item in value]
    return value


def _run_logged(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    redact_embedding_vectors: bool = False,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        log.write("COMMAND: " + json.dumps(command, ensure_ascii=False) + "\n\n")
        log.flush()
        try:
            if redact_embedding_vectors:
                completed = subprocess.run(
                    command,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    errors="replace",
                    check=False,
                )
                output = completed.stdout
                try:
                    output = json.dumps(
                        _redact_embeddings(json.loads(output)),
                        ensure_ascii=False,
                        indent=2,
                    )
                except json.JSONDecodeError:
                    pass
                log.write(output)
                if completed.stderr:
                    log.write("\nSTDERR:\n" + completed.stderr)
            else:
                completed = subprocess.run(
                    command,
                    cwd=cwd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
        except OSError as exc:
            log.write(f"\nCould not start command: {exc}\n")
            return 127
    return int(completed.returncode)


def _artifacts_missing(baseline_dir: Path, prediction_dir: Path) -> list[str]:
    required = [
        baseline_dir / "baseline.json",
        prediction_dir / "metadata.json",
        prediction_dir / "observations.jsonl",
        prediction_dir / "identity.jsonl",
        prediction_dir / "events.jsonl",
        prediction_dir / "event_lifecycle.jsonl",
        prediction_dir / "annotated.mp4",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if (baseline_dir / "baseline.json").is_file():
        try:
            baseline = read_json(baseline_dir / "baseline.json")
            reference_path = baseline_dir / str(
                (baseline.get("reference") or {}).get("image_path", "reference.png")
            )
            if not reference_path.is_file():
                missing.append(str(reference_path))
        except (OSError, ValueError, KeyError, TypeError):
            missing.append(str(baseline_dir / "reference.png"))
    if not (prediction_dir / "snapshots").is_dir():
        missing.append(str(prediction_dir / "snapshots"))
    video_path = prediction_dir / "annotated.mp4"
    if video_path.is_file() and video_path.stat().st_size == 0:
        missing.append(str(video_path) + " (empty)")
    return missing


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_sample(
    *,
    root: Path,
    run_root: Path,
    sample: Mapping[str, Any],
    annotation: Mapping[str, Any],
    template: Mapping[str, Any],
    state: dict[str, Any],
) -> None:
    sample_id = _safe_sample_id(sample)
    config, baseline_dir, prediction_dir = _build_sample_config(
        root, run_root, sample, annotation, template
    )
    config_path = run_root / "configs" / f"{sample_id}.json"
    write_json(config_path, config)
    entry = state["samples"][sample_id]
    entry.update({"status": "running", "started_at_utc": _utc_now(), "error": None})
    _atomic_json(run_root / "status.json", state)

    common = [sys.executable]
    calibrate_command = common + [
        str(root / "tools" / "calibrate.py"),
        "--config",
        _relative(root, config_path),
        "--output",
        _relative(root, baseline_dir),
        "--root",
        str(root),
    ]
    calibration_log = run_root / "logs" / sample_id / "calibration.log"
    calibration_code = _run_logged(
        calibrate_command,
        cwd=root,
        log_path=calibration_log,
        redact_embedding_vectors=True,
    )
    entry["calibration"] = {
        "exit_code": calibration_code,
        "log": _relative(root, calibration_log),
    }
    _atomic_json(run_root / "status.json", state)
    if calibration_code != 0:
        entry.update(
            {
                "status": "failed",
                "error": f"Calibration exited with code {calibration_code}",
                "finished_at_utc": _utc_now(),
            }
        )
        _atomic_json(run_root / "status.json", state)
        return

    detection_command = common + [
        str(root / "tools" / "run_change_detection.py"),
        "--config",
        _relative(root, config_path),
        "--output",
        _relative(root, prediction_dir),
        "--root",
        str(root),
    ]
    detection_log = run_root / "logs" / sample_id / "detection.log"
    detection_code = _run_logged(detection_command, cwd=root, log_path=detection_log)
    entry["detection"] = {
        "exit_code": detection_code,
        "log": _relative(root, detection_log),
    }
    missing = _artifacts_missing(baseline_dir, prediction_dir)
    if detection_code != 0 or missing:
        details = []
        if detection_code != 0:
            details.append(f"detection exited with code {detection_code}")
        if missing:
            details.append("missing artifacts: " + ", ".join(missing))
        entry.update(
            {
                "status": "failed",
                "error": "; ".join(details),
                "finished_at_utc": _utc_now(),
            }
        )
    else:
        entry.update(
            {
                "status": "completed",
                "finished_at_utc": _utc_now(),
                "artifacts": {
                    "baseline": _relative(root, baseline_dir),
                    "predictions": _relative(root, prediction_dir),
                    "config": _relative(root, config_path),
                },
            }
        )
    _atomic_json(run_root / "status.json", state)


def _evaluation_command(
    *,
    root: Path,
    run_root: Path,
    manifest_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    ignored_sample_ids: list[str],
) -> list[str]:
    command = [
        sys.executable,
        str(root / "tools" / "benchmark_dataset.py"),
        "--manifest",
        _relative(root, manifest_path),
        "--predictions-dir",
        _relative(root, run_root / "predictions"),
        "--root",
        str(root),
        "--split",
        args.split,
        "--output",
        _relative(root, output_path),
        "--event-boundary-tolerance",
        str(args.event_boundary_tolerance),
        "--forgotten-iou",
        str(args.forgotten_iou),
        "--moved-iou",
        str(args.moved_iou),
        "--latency-deadlines-seconds",
        *(str(value) for value in args.latency_deadlines_seconds),
    ]
    if args.require_object_id:
        command.append("--require-object-id")
    if args.split == "test":
        command.extend(["--allow-test", "--summary-only"])
    for sample_id in ignored_sample_ids:
        command.extend(["--ignore-predictions-for", sample_id])
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, default=Path("configs/m45.example.json"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/benchmark/v2/manifest.json"),
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Explicitly permit a blind run of the locked test split.",
    )
    parser.add_argument(
        "--test-lock",
        type=Path,
        default=Path("data/benchmark/v2/locked_test.json"),
        help="Lock receipt used to verify the locked test manifest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New batch directory under runs/ (defaults to a timestamped directory)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume this batch only if its manifest, inputs, config, and code still match.",
    )
    parser.add_argument("--event-boundary-tolerance", type=float, default=1.0)
    parser.add_argument("--forgotten-iou", type=float, default=0.3)
    parser.add_argument("--moved-iou", type=float, default=0.3)
    parser.add_argument("--require-object-id", action="store_true")
    parser.add_argument(
        "--latency-deadlines-seconds",
        type=float,
        nargs="+",
        default=(3.0, 5.0, 10.0),
    )
    parser.add_argument("--start-tolerance", type=float, default=3.0, help=argparse.SUPPRESS)
    parser.add_argument("--min-overlap", type=float, default=0.1, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        root = args.root.resolve()
        config_path = _inside_root(
            args.config if args.config.is_absolute() else root / args.config, root
        )
        manifest_path = _inside_root(
            args.manifest if args.manifest.is_absolute() else root / args.manifest,
            root,
        )
        if args.split == "test":
            if not args.allow_test:
                raise ValueError("Test execution is opt-in; pass --allow-test explicitly")
            lock_path = _inside_root(
                args.test_lock if args.test_lock.is_absolute() else root / args.test_lock,
                root,
            )
            _verify_test_lock(manifest_path, lock_path)
        template = read_json(config_path)
        manifest = read_json(manifest_path)
        samples = [
            item
            for item in manifest.get("samples", [])
            if str(item.get("split", "")) == args.split
        ]
        if not samples:
            raise ValueError(f"Manifest has no samples in split={args.split!r}")
        sample_ids = [_safe_sample_id(sample) for sample in samples]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError(f"Manifest contains duplicate video_id values in {args.split} split")

        annotations: dict[str, dict[str, Any]] = {}
        sample_fingerprints: dict[str, str] = {}
        template_sha256 = sha256_file(config_path)
        pipeline_sha256 = _pipeline_digest(root)
        for sample in samples:
            sample_id = _safe_sample_id(sample)
            annotation_path = resolve_repo_path(root, str(sample["annotation_path"]))
            annotation = read_json(annotation_path)
            annotations[sample_id] = annotation
            sample_fingerprints[sample_id] = _sample_fingerprint(
                root,
                sample,
                annotation,
                manifest=manifest,
                template_sha256=template_sha256,
                pipeline_sha256=pipeline_sha256,
            )

        if (
            not math.isfinite(args.event_boundary_tolerance)
            or args.event_boundary_tolerance < 0
        ):
            raise ValueError("event boundary tolerance must be finite and non-negative")
        deadlines = tuple(sorted(set(args.latency_deadlines_seconds)))
        if not deadlines or any(not math.isfinite(value) or value < 0 for value in deadlines):
            raise ValueError("latency deadlines must be finite and non-negative")
        args.latency_deadlines_seconds = deadlines
        evaluation_options = {
            "matching_policy": "confirmation_window",
            "event_boundary_tolerance_seconds": args.event_boundary_tolerance,
            "forgotten_iou_threshold": args.forgotten_iou,
            "moved_iou_threshold": args.moved_iou,
            "require_prediction_object_id": args.require_object_id,
            "latency_deadlines_seconds": list(deadlines),
        }
        run_fingerprint = _canonical_digest(
            {
                "manifest_sha256": sha256_file(manifest_path),
                "template_sha256": template_sha256,
                "pipeline_sha256": pipeline_sha256,
                "evaluation": evaluation_options,
                "split": args.split,
                "samples": sample_fingerprints,
            }
        )

        if args.output is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_root = root / "runs" / args.split / f"{config_path.stem}_all_{timestamp}"
        else:
            run_root = args.output if args.output.is_absolute() else root / args.output
        run_root = _inside_root(run_root, root)
        runs_root = (root / "runs").resolve()
        if run_root != runs_root and runs_root not in run_root.parents:
            raise ValueError("Batch output must be under the repository runs/ directory")
        status_path = run_root / "status.json"

        if run_root.exists():
            if not args.resume:
                raise FileExistsError(
                    f"Output directory already exists: {run_root}; choose a new path or use --resume"
                )
            if not status_path.is_file():
                raise ValueError(f"Cannot resume without {status_path}")
            state = read_json(status_path)
            if state.get("run_fingerprint") != run_fingerprint:
                raise ValueError(
                    "Cannot resume: manifest, media, config, evaluation settings, or code changed. "
                    "Choose a new output directory."
                )
        else:
            if args.resume:
                raise FileNotFoundError(f"Cannot resume; batch directory does not exist: {run_root}")
            run_root.mkdir(parents=True, exist_ok=False)
            state = {
                "run_id": run_root.name,
                "run_fingerprint": run_fingerprint,
                "created_at_utc": _utc_now(),
                "root": str(root),
                "interpreter": sys.executable,
                "manifest": _relative(root, manifest_path),
                "base_config": _relative(root, config_path),
                "split": args.split,
                "sample_count": len(samples),
                "annotation_policy": "Benchmark annotations are evaluated as stored; this tool does not edit labels.",
                "evaluation": evaluation_options,
                "samples": {
                    _safe_sample_id(sample): {
                        "status": "pending",
                        "fingerprint": sample_fingerprints[_safe_sample_id(sample)],
                    }
                    for sample in samples
                },
            }
            _atomic_json(status_path, state)

        for index, sample in enumerate(samples, start=1):
            sample_id = _safe_sample_id(sample)
            entry = state["samples"][sample_id]
            label = sample_id if args.split == "validation" else "blind sample"
            print(f"[{index}/{len(samples)}] {label}", flush=True)
            try:
                _, baseline_dir, prediction_dir = _build_sample_config(
                    root, run_root, sample, annotations[sample_id], template
                )
                resume_artifacts_present = all(
                    path.is_file()
                    for path in (
                        run_root / "configs" / f"{sample_id}.json",
                        run_root / "logs" / sample_id / "calibration.log",
                        run_root / "logs" / sample_id / "detection.log",
                    )
                )
                if (
                    entry.get("status") == "completed"
                    and entry.get("fingerprint") == sample_fingerprints[sample_id]
                    and not _artifacts_missing(baseline_dir, prediction_dir)
                    and resume_artifacts_present
                ):
                    print("  already complete; skipping", flush=True)
                    continue
                _run_sample(
                    root=root,
                    run_root=run_root,
                    sample=sample,
                    annotation=annotations[sample_id],
                    template=template,
                    state=state,
                )
                if args.split == "test":
                    print(f"  {entry.get('status')}", flush=True)
                else:
                    print(f"  {entry.get('status')}: {entry.get('error') or 'done'}", flush=True)
            except Exception as exc:  # keep the batch moving and report this sample
                entry.update(
                    {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "finished_at_utc": _utc_now(),
                    }
                )
                _atomic_json(status_path, state)
                if args.split == "test":
                    print("  failed; details are sealed in the run status", flush=True)
                else:
                    print(f"  failed: {type(exc).__name__}: {exc}", flush=True)

        report_path = run_root / "metrics" / "evaluation.json"
        ignored_sample_ids = [
            sample_id
            for sample_id, entry in state["samples"].items()
            if entry.get("status") != "completed"
        ]
        log_path = run_root / "logs" / "evaluate.log"
        command = _evaluation_command(
            root=root,
            run_root=run_root,
            manifest_path=manifest_path,
            output_path=report_path,
            args=args,
            ignored_sample_ids=ignored_sample_ids,
        )
        code = _run_logged(command, cwd=root, log_path=log_path)
        evaluation_states = {
            "evaluation": {
                "exit_code": code,
                "report": _relative(root, report_path),
                "log": _relative(root, log_path),
            }
        }
        _atomic_json(status_path, state)

        failed = [
            sample_id
            for sample_id, entry in state["samples"].items()
            if entry.get("status") != "completed"
        ]
        report_failures = [
            key for key, value in evaluation_states.items() if value["exit_code"] != 0
        ]
        summary = {
            "run_id": state["run_id"],
            "run_fingerprint": run_fingerprint,
            "split": args.split,
            "sample_count": len(samples),
            "completed_count": len(samples) - len(failed),
            "failed_count": len(failed),
            "failed_samples": failed if args.split == "validation" else [],
            "evaluation_failures": report_failures,
            "evaluation_reports": evaluation_states,
            "latency_deadlines_seconds": list(deadlines),
            "annotation_policy": state.get("annotation_policy"),
            "finished_at_utc": _utc_now(),
        }
        if args.split == "test" and code == 0 and report_path.is_file():
            summary["metrics"] = read_json(report_path).get("overall", {})
        write_json(run_root / "summary.json", summary)
        state["finished_at_utc"] = summary["finished_at_utc"]
        state["evaluation_reports"] = evaluation_states
        _atomic_json(status_path, state)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1 if failed or report_failures else 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        prefix = "Blind test" if getattr(args, "split", "validation") == "test" else "Validation"
        print(f"{prefix} batch failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
