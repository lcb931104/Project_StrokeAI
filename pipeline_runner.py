from __future__ import annotations

import argparse
import json
import queue
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pipeline_common import (
    ATLAS_NAMES,
    ensure_file,
    get_config,
    nii_basename,
    prepare_nnunet_input,
    require_modules,
    save_reorient_mat,
)


LogFn = Callable[[str], None]


@dataclass
class PipelineOptions:
    spm_runner: str = "standalone"
    dataset_id: str = "2"
    configuration: str = "3d_fullres"
    fold: str = "1"
    trainer: str = "nnUNetTrainer_300epoch"
    device: str = "auto"
    sequential: bool = False


def format_cmd(cmd: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in cmd)


def stream_process(name: str, cmd: list[str], log: LogFn) -> int:
    log(f"[{name}] Running: {format_cmd(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log(f"[{name}] {line.rstrip()}")
    code = proc.wait()
    log(f"[{name}] Exit code: {code}")
    return code


def run_parallel_steps(step2: list[str], step3: list[str], log: LogFn) -> tuple[int, int]:
    results: dict[str, int] = {}

    def worker(name: str, cmd: list[str]):
        results[name] = stream_process(name, cmd, log)

    t2 = threading.Thread(target=worker, args=("step2-nnunet", step2), daemon=True)
    t3 = threading.Thread(target=worker, args=("step3-spm", step3), daemon=True)
    log("Running steps 2 and 3 in parallel...")
    t2.start()
    t3.start()
    t2.join()
    t3.join()
    return results.get("step2-nnunet", 1), results.get("step3-spm", 1)


def run_clinical_symptom_step(project_dir: Path, structures: list[str], log: LogFn) -> str:
    if not structures:
        return "No quantified regions."
    cmd = [sys.executable, str(project_dir / "6_clinical_symptom.py")]
    log("[step6-clinical] Running clinical symptom lookup...")
    proc = subprocess.run(
        cmd,
        input="\n".join(structures) + "\n",
        text=True,
        capture_output=True,
    )
    if proc.stdout:
        for line in proc.stdout.splitlines():
            log(f"[step6-clinical] {line}")
    if proc.stderr:
        for line in proc.stderr.splitlines():
            log(f"[step6-clinical] {line}")
    if proc.returncode:
        raise RuntimeError(f"step6-clinical failed with exit code {proc.returncode}")
    return proc.stdout.strip()


def unique_work_dir(input_nii: Path) -> Path:
    base_dir = input_nii.parent / f"{nii_basename(input_nii)}_work"
    if not base_dir.exists():
        return base_dir
    index = 1
    while True:
        candidate = input_nii.parent / f"{nii_basename(input_nii)}_work({index})"
        if not candidate.exists():
            return candidate
        index += 1


def prepare_work_input(input_nii: Path) -> Path:
    work_dir = unique_work_dir(input_nii)
    work_dir.mkdir(parents=True)
    copied_input = work_dir / input_nii.name
    shutil.copy2(input_nii, copied_input)
    return copied_input


def run_pipeline(input_nii: Path, options: PipelineOptions, log: LogFn = print) -> tuple[list[dict], str, Path]:
    require_modules()
    project_dir = Path(__file__).resolve().parent
    input_nii = input_nii.expanduser().resolve()
    ensure_file(input_nii, "Input NIfTI")
    input_nii = prepare_work_input(input_nii)
    log(f"Working folder: {input_nii.parent}")
    log(f"Copied input NIfTI: {input_nii}")

    save_reorient_mat(input_nii)
    nnunet_input, _ = prepare_nnunet_input(input_nii, input_nii.parent / "image")
    log(f"Prepared nnU-Net input: {nnunet_input}")

    step2 = [
        sys.executable,
        str(project_dir / "2_nnunet_predict.py"),
        str(input_nii),
        "--dataset-id",
        options.dataset_id,
        "--configuration",
        options.configuration,
        "--fold",
        options.fold,
        "--trainer",
        options.trainer,
        "--device",
        options.device,
    ]
    step3 = [
        sys.executable,
        str(project_dir / "3_spm_make_transform.py"),
        str(input_nii),
        "--spm-runner",
        options.spm_runner,
    ]

    if options.sequential:
        for name, cmd in (("step2-nnunet", step2), ("step3-spm", step3)):
            code = stream_process(name, cmd, log)
            if code:
                raise RuntimeError(f"{name} failed with exit code {code}")
    else:
        code2, code3 = run_parallel_steps(step2, step3, log)
        if code2 or code3:
            raise RuntimeError(f"Parallel steps failed: step2={code2}, step3={code3}")

    result_json = input_nii.parent / "result" / f"{input_nii.stem}_analysis.json"
    step4 = [
        sys.executable,
        str(project_dir / "4_normalize_and_analyze.py"),
        str(input_nii),
        "--spm-runner",
        options.spm_runner,
        "--result-json",
        str(result_json),
    ]
    code4 = stream_process("step4-analysis", step4, log)
    if code4:
        raise RuntimeError(f"step4-analysis failed with exit code {code4}")

    results: list[dict] = []
    if result_json.is_file():
        results = json.loads(result_json.read_text(encoding="utf-8"))
        log("")
        log("Final results returned to step 1:")
        for row in results:
            log(
                f"{row['structure']}  Distribution: {row['region']:.4f}%  "
                f"Involvement: {row['lesion']:.4f}%  Volume: {row['volume_ml']:.4f} mL"
            )

    clinical_output = run_clinical_symptom_step(
        project_dir,
        [row["structure"] for row in results],
        log,
    )

    step5 = [
        sys.executable,
        str(project_dir / "5_postprocessing.py"),
        str(input_nii),
    ]
    code5 = stream_process("step5-postprocessing", step5, log)
    if code5:
        raise RuntimeError(f"step5-postprocessing failed with exit code {code5}")

    return results, clinical_output, input_nii

# 20260721
