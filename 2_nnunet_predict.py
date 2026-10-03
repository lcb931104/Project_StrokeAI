from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_common import (
    align_nifti_to_reference_space,
    ensure_file,
    get_config,
    gunzip_prediction,
    prepare_nnunet_input,
    run_nnunet,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 2: run nnU-Net infarct segmentation.")
    parser.add_argument("input_nii", type=Path)
    parser.add_argument("--dataset-id", default="2")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--fold", default="1")
    parser.add_argument("--trainer", default="nnUNetTrainer_300epoch")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another nnU-Net device value.")
    parser.add_argument("--skip-if-exists", action="store_true")
    args = parser.parse_args()

    cfg = get_config()
    input_nii = args.input_nii.expanduser().resolve()
    ensure_file(input_nii, "Input NIfTI")

    input_dir = input_nii.parent
    image_dir = input_dir / "image"
    result_dir = input_dir / "result"
    _, case_name = prepare_nnunet_input(input_nii, image_dir)

    label_nii = result_dir / f"{case_name}.nii"
    if args.skip_if_exists and label_nii.is_file():
        align_nifti_to_reference_space(input_nii, label_nii)
        print(f"nnU-Net result already exists: {label_nii}")
        return 0

    run_nnunet(image_dir, result_dir, args, cfg)
    label_nii = gunzip_prediction(result_dir, case_name)
    align_nifti_to_reference_space(input_nii, label_nii)
    print(f"nnU-Net label: {label_nii}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# 20260721
