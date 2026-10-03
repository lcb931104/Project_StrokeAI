from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pipeline_common import ensure_file, nii_basename


class SegmentationFailedError(RuntimeError):
    pass


def patient_id_from_input(input_nii: Path) -> str:
    return input_nii.parent.name


def build_targets(input_nii: Path) -> dict[str, tuple[Path, Path]]:
    patient_dir = input_nii.parent
    patient_id = patient_id_from_input(input_nii)
    case_name = nii_basename(input_nii)

    return {
        "dwi_before": (
            patient_dir / f"{case_name}.nii",
            patient_dir / f"{case_name}_before.nii",
        ),
        "dwi_after": (
            patient_dir / f"w{case_name}.nii",
            patient_dir / f"{case_name}_after.nii",
        ),
        "mask_before": (
            patient_dir / "result" / f"{case_name}.nii",
            patient_dir / f"{patient_id}_mask_before.nii",
        ),
        "mask_after": (
            patient_dir / "result" / f"w{case_name}.nii",
            patient_dir / f"{patient_id}_mask_after.nii",
        ),
        "mask_atlas": (
            patient_dir / "result" / f"{patient_id}_mask_atlas.nii",
            patient_dir / f"{patient_id}_mask_atlas.nii",
        ),
    }


def delete_existing_final_files(rename_map: dict[str, tuple[Path, Path]], dry_run: bool):
    for _, target in rename_map.values():
        if not target.exists():
            continue
        if dry_run:
            print(f"Would delete existing final file: {target}")
            continue
        print(f"Delete existing final file: {target}")
        target.unlink()


def validate_sources(rename_map: dict[str, tuple[Path, Path]]):
    for key in ("dwi_before", "dwi_after"):
        source, _ = rename_map[key]
        ensure_file(source, "Required DWI output")

    missing_masks = []
    for key in ("mask_before", "mask_after", "mask_atlas"):
        source, _ = rename_map[key]
        if not source.is_file():
            missing_masks.append(source)

    if missing_masks:
        missing = "\n".join(str(path) for path in missing_masks)
        raise SegmentationFailedError(f"分割失敗：找不到 mask 輸出檔\n{missing}")


def move_final_files(rename_map: dict[str, tuple[Path, Path]]):
    for source, target in rename_map.values():
        print(f"Move: {source} -> {target}")
        shutil.move(str(source), str(target))


def cleanup_patient_dir(input_nii: Path, keep_files: set[Path], dry_run: bool):
    patient_dir = input_nii.parent
    for item in patient_dir.iterdir():
        if item.resolve() in keep_files:
            continue

        if dry_run:
            print(f"Would delete: {item}")
            continue

        print(f"Delete: {item}")
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def run_postprocessing(input_nii: Path, dry_run: bool = False):
    input_nii = input_nii.expanduser().resolve()
    patient_dir = input_nii.parent
    if not patient_dir.is_dir():
        raise NotADirectoryError(f"Patient folder not found: {patient_dir}")

    rename_map = build_targets(input_nii)

    validate_sources(rename_map)
    delete_existing_final_files(rename_map, dry_run=dry_run)

    final_files = {target.resolve() for _, target in rename_map.values()}

    if dry_run:
        for source, target in rename_map.values():
            print(f"Would move: {source} -> {target}")
        dry_run_keep = final_files | {source.resolve() for source, _ in rename_map.values()}
        cleanup_patient_dir(input_nii, dry_run_keep, dry_run=True)
        return

    move_final_files(rename_map)
    cleanup_patient_dir(input_nii, final_files, dry_run=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 5: rename the final NIfTI outputs and delete intermediate files."
    )
    parser.add_argument("input_nii", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_postprocessing(args.input_nii, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# 20260721