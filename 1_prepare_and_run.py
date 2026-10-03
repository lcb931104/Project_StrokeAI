from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline_common import (
    convert_dicom_files_to_nifti,
    dicom_files_in_folder,
    dicom_nifti_output_from_header,
    is_nifti_file,
    read_dicom_header,
)
from pipeline_runner import PipelineOptions, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 1: prepare NIfTI/DICOM data, run steps 2 and 3, then run steps 4, 6, and 5."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        help="Input NIfTI file, DICOM file, or folder containing one DICOM series.",
    )
    parser.add_argument(
        "--dicom-files",
        nargs="+",
        type=Path,
        help="Specific DICOM files to convert into one NIfTI before running the pipeline.",
    )
    parser.add_argument("--spm-runner", choices=["standalone", "matlab"], default="standalone")
    parser.add_argument("--dataset-id", default="2")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--fold", default="1")
    parser.add_argument("--trainer", default="nnUNetTrainer_300epoch")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another nnU-Net device value.")
    parser.add_argument("--sequential", action="store_true", help="Run steps 2 and 3 one after another.")
    return parser.parse_args()


def resolve_pipeline_input(args: argparse.Namespace) -> Path:
    if args.dicom_files:
        header = read_dicom_header(args.dicom_files)
        output_file = dicom_nifti_output_from_header(args.dicom_files[0].parent, header)
        return convert_dicom_files_to_nifti(args.dicom_files, output_file)
    if args.input_path is None:
        raise ValueError("Provide a NIfTI input path, a DICOM folder/file, or --dicom-files.")

    input_path = args.input_path.expanduser().resolve()
    if input_path.is_dir():
        dicom_files = dicom_files_in_folder(input_path)
        header = read_dicom_header(dicom_files)
        output_file = dicom_nifti_output_from_header(input_path, header)
        return convert_dicom_files_to_nifti(dicom_files, output_file)
    if is_nifti_file(input_path):
        return input_path
    header = read_dicom_header([input_path])
    output_file = dicom_nifti_output_from_header(input_path.parent, header)
    return convert_dicom_files_to_nifti([input_path], output_file)


def main() -> int:
    args = parse_args()
    options = PipelineOptions(
        spm_runner=args.spm_runner,
        dataset_id=args.dataset_id,
        configuration=args.configuration,
        fold=args.fold,
        trainer=args.trainer,
        device=args.device,
        sequential=args.sequential,
    )

    try:
        run_pipeline(resolve_pipeline_input(args), options)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# 20260721
