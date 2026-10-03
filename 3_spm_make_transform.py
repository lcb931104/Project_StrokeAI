from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from pipeline_common import ensure_file, find_sn_mat, get_config, run_spm_batch, write_segment_batch


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 3: run SPM Old Segment to create the standard-space transform matrix."
    )
    parser.add_argument("input_nii", type=Path)
    parser.add_argument("--spm-runner", choices=["standalone", "matlab"], default="standalone")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = get_config()
    input_nii = args.input_nii.expanduser().resolve()
    ensure_file(input_nii, "Input NIfTI")

    if not args.force:
        try:
            sn_mat = find_sn_mat(input_nii.parent)
            print(f"SPM transform already exists: {sn_mat}")
            return 0
        except FileNotFoundError:
            pass

    with tempfile.TemporaryDirectory(prefix="spm_segment_") as tmp:
        batch_file = Path(tmp) / "segment_batch.m"
        write_segment_batch(batch_file, input_nii)
        run_spm_batch(batch_file, args.spm_runner, cfg)

    sn_mat = find_sn_mat(input_nii.parent)
    print(f"SPM transform: {sn_mat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# 20260721