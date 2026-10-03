from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from pipeline_common import (
    align_nifti_to_reference_space,
    atlas_center_mm,
    calculate_overlap,
    ensure_file,
    find_sn_mat,
    get_config,
    native_mask_volume_ml,
    nii_basename,
    run_spm_batch,
    set_image_center_to_atlas,
    write_normalize_batch,
    write_mask_atlas_copy,
)


def filtered_rows(rows, total_volume_ml: float | None = None):
    results = []
    for name, region, lesion, absolute in rows:
        if region >= 5.0 or lesion >= 10.0 or absolute >= 0.12:
            row = {
                "structure": name,
                "region": region,
                "lesion": lesion,
                "volume_ml": absolute,
            }
            if total_volume_ml is not None:
                row["total_volume_ml"] = total_volume_ml
            results.append(row)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 4: normalize segmentation to standard space and analyze atlas overlap."
    )
    parser.add_argument("input_nii", type=Path)
    parser.add_argument("--spm-runner", choices=["standalone", "matlab"], default="standalone")
    parser.add_argument("--skip-normalize", action="store_true")
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()

    cfg = get_config()
    input_nii = args.input_nii.expanduser().resolve()
    ensure_file(input_nii, "Input NIfTI")
    ensure_file(cfg.atlas_file, "Atlas")

    case_name = nii_basename(input_nii)
    label_nii = input_nii.parent / "result" / f"{case_name}.nii"
    ensure_file(label_nii, "nnU-Net label")
    align_nifti_to_reference_space(input_nii, label_nii)

    sn_mat = find_sn_mat(input_nii.parent)

    if not args.skip_normalize:
        with tempfile.TemporaryDirectory(prefix="spm_normalize_") as tmp:
            batch_file = Path(tmp) / "normalize_batch.m"
            write_normalize_batch(batch_file, sn_mat, [label_nii, input_nii])
            run_spm_batch(batch_file, args.spm_runner, cfg)

    w_label = label_nii.with_name("w" + label_nii.name)
    w_data = input_nii.with_name("w" + input_nii.name)
    ensure_file(w_label, "Normalized label")
    ensure_file(w_data, "Normalized input image")

    center_mm = atlas_center_mm(cfg.atlas_file)
    set_image_center_to_atlas(w_label, center_mm)
    set_image_center_to_atlas(w_data, center_mm)
    mask_atlas = label_nii.with_name(f"{input_nii.parent.name}_mask_atlas.nii")
    write_mask_atlas_copy(cfg.atlas_file, w_label, mask_atlas)
    print(f"Saved mask atlas: {mask_atlas}")

    total_volume_ml = native_mask_volume_ml(input_nii, label_nii)
    results = filtered_rows(
        calculate_overlap(
            cfg.atlas_file,
            w_label,
            native_image_file=input_nii,
            native_mask_file=label_nii,
        ),
        total_volume_ml=total_volume_ml,
    )
    for row in results:
        print(
            f"{row['structure']}  Distribution: {row['region']:.4f}%  "
            f"Involvement: {row['lesion']:.4f}%  Volume: {row['volume_ml']:.4f} mL"
        )

    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Saved result JSON: {args.result_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# 20260721
