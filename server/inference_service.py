"""
Inference service module for managing HoVerNet inference jobs.

Handles:
- Extracting ROI regions from WSI slides
- Launching inference as subprocess in the 'hovernet' conda env
- Tracking job status and progress
- Storing and retrieving results
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


class JobStatus(str, Enum):
    PENDING = "pending"
    EXTRACTING = "extracting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class InferenceJob:
    job_id: str
    slide_id: str
    model_id: str
    status: JobStatus = JobStatus.PENDING
    progress: int = 0
    status_message: str = "Queued"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    result_path: Optional[str] = None
    roi: Optional[dict] = None  # {x, y, width, height} in level-0 coords
    image_path: Optional[str] = None
    process: Optional[subprocess.Popen] = None
    _result_cache: Optional[dict] = None

    @property
    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0
        end = self.completed_at or time.time()
        return round(end - self.started_at, 1)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "slide_id": self.slide_id,
            "model_id": self.model_id,
            "status": self.status.value,
            "progress": self.progress,
            "status_message": self.status_message,
            "elapsed_seconds": self.elapsed_seconds,
            "roi": self.roi,
            "error": self.error,
        }

    def get_results(self) -> Optional[dict]:
        if self._result_cache is not None:
            return self._result_cache
        if self.result_path and os.path.exists(self.result_path):
            with open(self.result_path, "r") as f:
                self._result_cache = json.load(f)
            return self._result_cache
        return None


class InferenceJobManager:
    """Manages inference jobs for all slides."""

    def __init__(self, project_root: Path, conda_env: str = "hovernet"):
        self.project_root = project_root
        self.conda_env = conda_env
        self.results_dir = project_root / "data" / "inference_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, InferenceJob] = {}
        self._lock = threading.Lock()

        # Paths
        self.runner_script = project_root / "server" / "hovernet_runner.py"
        self.model_path = project_root / "hover_net" / "hovernet_fast_pannuke_type_tf2pytorch.tar"
        self.type_info_path = project_root / "hover_net" / "type_info.json"

    def start_inference(
        self,
        slide_entry,  # SlideEntry from the main app
        model_id: str = "hovernet",
        roi: Optional[dict] = None,
        device: str = "auto",
    ) -> InferenceJob:
        """Start an inference job for a slide region.

        Args:
            slide_entry: SlideEntry object with slide info
            model_id: model identifier
            roi: optional region of interest {x, y, width, height} in level-0 coords
            device: 'auto', 'mps', or 'cpu'
        """
        job_id = uuid.uuid4().hex[:12]
        job = InferenceJob(
            job_id=job_id,
            slide_id=slide_entry.slide_id,
            model_id=model_id,
            roi=roi,
        )

        with self._lock:
            self.jobs[job_id] = job

        # Start in background thread
        thread = threading.Thread(
            target=self._run_job,
            args=(job, slide_entry, device),
            daemon=True,
        )
        thread.start()

        return job

    def get_job(self, job_id: str) -> Optional[InferenceJob]:
        return self.jobs.get(job_id)

    def get_jobs_for_slide(self, slide_id: str) -> list[InferenceJob]:
        return [j for j in self.jobs.values() if j.slide_id == slide_id]

    def get_latest_completed_job(self, slide_id: str) -> Optional[InferenceJob]:
        completed = [
            j for j in self.jobs.values()
            if j.slide_id == slide_id and j.status == JobStatus.COMPLETED
        ]
        if not completed:
            return None
        return max(completed, key=lambda j: j.completed_at or 0)

    def cancel_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job and job.status in (JobStatus.PENDING, JobStatus.RUNNING, JobStatus.EXTRACTING):
            job.status = JobStatus.CANCELLED
            job.status_message = "Cancelled by user"
            if job.process:
                try:
                    job.process.terminate()
                except Exception:
                    pass
            return True
        return False

    def _run_job(self, job: InferenceJob, slide_entry, device: str):
        """Execute the inference pipeline in a background thread."""
        try:
            job.started_at = time.time()
            job.status = JobStatus.EXTRACTING
            job.status_message = "Extracting region from slide..."
            job.progress = 2

            # Create job output directory
            job_dir = self.results_dir / job.slide_id / job.job_id
            job_dir.mkdir(parents=True, exist_ok=True)

            # Extract region from slide
            image_path = job_dir / "region.png"
            self._extract_region(slide_entry, job.roi, image_path)
            job.image_path = str(image_path)
            job.progress = 10

            if job.status == JobStatus.CANCELLED:
                return

            # Run HoVerNet inference via subprocess
            job.status = JobStatus.RUNNING
            job.status_message = "Loading model..."
            job.progress = 12

            result_path = job_dir / "results.json"
            job.result_path = str(result_path)

            self._run_hovernet_subprocess(job, str(image_path), str(result_path), device)

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.status_message = f"Error: {e}"
            job.completed_at = time.time()

    def _extract_region(self, slide_entry, roi: Optional[dict], output_path: Path):
        """Extract a region from the slide and save as PNG."""
        slide = slide_entry.slide

        if roi:
            x = int(roi["x"])
            y = int(roi["y"])
            w = int(roi["width"])
            h = int(roi["height"])
            # Clamp to slide dimensions
            sw, sh = slide.dimensions
            x = max(0, min(x, sw - 1))
            y = max(0, min(y, sh - 1))
            w = min(w, sw - x)
            h = min(h, sh - y)

            # If region is too large, downsample using a lower level
            # Max reasonable size for inference: ~4000x4000 pixels
            max_dim = 4000
            if w > max_dim or h > max_dim:
                # Find appropriate level
                scale = max(w, h) / max_dim
                level = 0
                for lvl in range(slide.level_count):
                    ds = slide.level_downsamples[lvl]
                    if ds >= scale:
                        level = lvl
                        break
                    level = lvl
                ds = slide.level_downsamples[level]
                read_size = (int(w / ds), int(h / ds))
                region = slide.read_region((x, y), level, read_size)
            else:
                region = slide.read_region((x, y), 0, (w, h))

            # Convert RGBA to RGB
            region = region.convert("RGB")
        else:
            # No ROI - extract thumbnail at a reasonable resolution
            # Use level that gives ~2000px on the longest side
            sw, sh = slide.dimensions
            max_dim = 2000
            scale = max(sw, sh) / max_dim
            level = 0
            for lvl in range(slide.level_count):
                ds = slide.level_downsamples[lvl]
                if ds >= scale:
                    level = lvl
                    break
                level = lvl
            dims = slide.level_dimensions[level]
            region = slide.read_region((0, 0), level, dims)
            region = region.convert("RGB")

        region.save(str(output_path), "PNG")

    def _run_hovernet_subprocess(self, job: InferenceJob, image_path: str,
                                  result_path: str, device: str):
        """Run HoVerNet via subprocess in the hovernet conda env."""
        cmd = [
            "conda", "run", "-n", self.conda_env, "--no-capture-output",
            "python", str(self.runner_script),
            "--image_path", image_path,
            "--output_path", result_path,
            "--model_path", str(self.model_path),
            "--type_info_path", str(self.type_info_path),
            "--nr_types", "6",
            "--batch_size", "8",
            "--device", device,
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(self.project_root),
        )
        job.process = process

        # Read stdout for progress updates
        for line in iter(process.stdout.readline, ""):
            if job.status == JobStatus.CANCELLED:
                process.terminate()
                return

            line = line.strip()
            if line.startswith("PROGRESS:"):
                try:
                    job.progress = int(line.split(":")[1])
                except ValueError:
                    pass
            elif line.startswith("STATUS:"):
                job.status_message = line.split(":", 1)[1]

        process.wait()

        if job.status == JobStatus.CANCELLED:
            return

        if process.returncode != 0:
            stderr = process.stderr.read() if process.stderr else ""
            job.status = JobStatus.FAILED
            job.error = stderr[-500:] if stderr else "Process exited with non-zero code"
            job.status_message = "Inference failed"
            job.completed_at = time.time()
            return

        # Check result file
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                result = json.load(f)
            if result.get("status") == "success":
                job.status = JobStatus.COMPLETED
                job.progress = 100
                count = result.get("nuclei_count", 0)
                job.status_message = f"Complete! {count} nuclei detected"
            else:
                job.status = JobStatus.FAILED
                job.error = result.get("error", "Unknown error")
                job.status_message = "Inference failed"
        else:
            job.status = JobStatus.FAILED
            job.error = "Result file not created"
            job.status_message = "Inference failed"

        job.completed_at = time.time()
