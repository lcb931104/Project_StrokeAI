from __future__ import annotations

import gzip
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ATLAS_NAMES = [
    "SUPERIOR PARIETAL LOBULE (left)",
    "CINGULATE GYRUS (left)",
    "SUPERIOR FRONTAL GYRUS (left)",
    "MIDDLE FRONTAL GYRUS (left)",
    "INFERIOR FRONTAL GYRUS (left)",
    "PRECENTRAL GYRUS (left)",
    "POSTCENTRAL GYRUS (left)",
    "ANGULAR GYRUS (left)",
    "PRECUNEUS (left)",
    "CUNEUS (left)",
    "LINGUAL GYRUS (left)",
    "FUSIFORM GYRUS (left)",
    "PARAHIPPOCAMPAL GYRUS (left)",
    "SUPERIOR OCCIPITAL GYRUS (left)",
    "INFERIOR OCCIPITAL GYRUS (left)",
    "MIDDLE OCCIPITAL GYRUS (left)",
    "ENTORHINAL AREA (left)",
    "SUPERIOR TEMPORAL GYRUS (left)",
    "INFERIOR TEMPORAL GYRUS (left)",
    "MIDDLE TEMPORAL GYRUS (left)",
    "LATERAL FRONTO-ORBITAL GYRUS (left)",
    "MIDDLE FRONTO-ORBITAL GYRUS (left)",
    "SUPRAMARGINAL GYRUS (left)",
    "GYRUS RECTUS (left)",
    "INSULA (left)",
    "AMYGDALA (left)",
    "HIPPOCAMPUS (left)",
    "CEREBELLUM (left)",
    "CORTICOSPINAL TRACT (left)",
    "INFERIOR CEREBELLAR PEDUNCLE (left)",
    "MEDIAL LEMNISCUS (left)",
    "SUPERIOR CEREBELLAR PEDUNCLE (left)",
    "CEREBRAL PEDUNCLE (left)",
    "ANTERIOR LIMB OF INTERNAL CAPSULE (left)",
    "POSTERIOR LIMB OF INTERNAL CAPSULE (left)",
    "POSTERIOR THALAMIC RADIATION (left)",
    "ANTERIOR CORONA RADIATA (left)",
    "SUPERIOR CORONA RADIATA (left)",
    "POSTERIOR CORONA RADIATA (left)",
    "CINGULUM (cingulate gyrus, left)",
    "CINGULUM (hippocampus, left)",
    "FORNIX (cres)/STRIA TERMINALIS (left)",
    "SUPERIOR LONGITUDINAL FASCICULUS (left)",
    "SUPERIOR FRONTO-OCCIPITAL FASCICULUS (left)",
    "INFERIOR FRONTO-OCCIPITAL FASCICULUS (left)",
    "SAGITTAL STRATUM (left)",
    "EXTERNAL CAPSULE (left)",
    "UNCINATE FASCICULUS (left)",
    "PONTINE CROSSING TRACT (left)",
    "MIDDLE CEREBELLAR PEDUNCLE (left)",
    "FORNIX (column and body, left)",
    "GENU OF CORPUS CALLOSUM (left)",
    "BODY OF CORPUS CALLOSUM (left)",
    "SPLENIUM OF CORPUS CALLOSUM (left)",
    "RETROLENTICULAR PART OF INTERNAL CAPSULE (left)",
    "RED NUCLEUS (left)",
    "SUBSTANTIA NIGRA (left)",
    "TAPETUM (left)",
    "CAUDATE NUCLEUS (left)",
    "PUTAMEN (left)",
    "THALAMUS (left)",
    "GLOBUS PALLIDUS (left)",
    "MIDBRAIN (left)",
    "PONS (left)",
    "MEDULLA (left)",
    "SUPERIOR PARIETAL LOBULE (right)",
    "CINGULATE GYRUS (right)",
    "SUPERIOR FRONTAL GYRUS (right)",
    "MIDDLE FRONTAL GYRUS (right)",
    "INFERIOR FRONTAL GYRUS (right)",
    "PRECENTRAL GYRUS (right)",
    "POSTCENTRAL GYRUS (right)",
    "ANGULAR GYRUS (right)",
    "PRECUNEUS (right)",
    "CUNEUS (right)",
    "LINGUAL GYRUS (right)",
    "FUSIFORM GYRUS (right)",
    "PARAHIPPOCAMPAL GYRUS (right)",
    "SUPERIOR OCCIPITAL GYRUS (right)",
    "INFERIOR OCCIPITAL GYRUS (right)",
    "MIDDLE OCCIPITAL GYRUS (right)",
    "ENTORHINAL AREA (right)",
    "SUPERIOR TEMPORAL GYRUS (right)",
    "INFERIOR TEMPORAL GYRUS (right)",
    "MIDDLE TEMPORAL GYRUS (right)",
    "LATERAL FRONTO-ORBITAL GYRUS (right)",
    "MIDDLE FRONTO-ORBITAL GYRUS (right)",
    "SUPRAMARGINAL GYRUS (right)",
    "GYRUS RECTUS (right)",
    "INSULA (right)",
    "AMYGDALA (right)",
    "HIPPOCAMPUS (right)",
    "CEREBELLUM (right)",
    "CORTICOSPINAL TRACT (right)",
    "INFERIOR CEREBELLAR PEDUNCLE (right)",
    "MEDIAL LEMNISCUS (right)",
    "SUPERIOR CEREBELLAR PEDUNCLE (right)",
    "CEREBRAL PEDUNCLE (right)",
    "ANTERIOR LIMB OF INTERNAL CAPSULE (right)",
    "POSTERIOR LIMB OF INTERNAL CAPSULE (right)",
    "POSTERIOR THALAMIC RADIATION (right)",
    "ANTERIOR CORONA RADIATA (right)",
    "SUPERIOR CORONA RADIATA (right)",
    "POSTERIOR CORONA RADIATA (right)",
    "CINGULUM (cingulate gyrus, right)",
    "CINGULUM (hippocampus, right)",
    "FORNIX (cres)/STRIA TERMINALIS (right)",
    "SUPERIOR LONGITUDINAL FASCICULUS (right)",
    "SUPERIOR FRONTO-OCCIPITAL FASCICULUS (right)",
    "INFERIOR FRONTO-OCCIPITAL FASCICULUS (right)",
    "SAGITTAL STRATUM (right)",
    "EXTERNAL CAPSULE (right)",
    "UNCINATE FASCICULUS (right)",
    "PONTINE CROSSING TRACT (right)",
    "MIDDLE CEREBELLAR PEDUNCLE (right)",
    "FORNIX (column and body, right)",
    "GENU OF CORPUS CALLOSUM (right)",
    "BODY OF CORPUS CALLOSUM (right)",
    "SPLENIUM OF CORPUS CALLOSUM (right)",
    "RETROLENTICULAR PART OF INTERNAL CAPSULE (right)",
    "RED NUCLEUS (right)",
    "SUBSTANTIA NIGRA (right)",
    "TAPETUM (right)",
    "CAUDATE NUCLEUS (right)",
    "PUTAMEN (right)",
    "THALAMUS (right)",
    "GLOBUS PALLIDUS (right)",
    "MIDBRAIN (right)",
    "PONS (right)",
    "MEDULLA (right)",
]


@dataclass(frozen=True)
class Config:
    project_dir: Path
    nnunet_predict: Path
    spm12_exe: Path | None
    mcr_root: Path | None
    matlab_exe: str
    atlas_file: Path
    nnunet_raw: Path
    nnunet_preprocessed: Path
    nnunet_results: Path


def get_config() -> Config:
    project_dir = Path(__file__).resolve().parent
    spm_candidates = [
        project_dir / "spm12_r7771_R2010a" / "spm12" / "spm12_win64.exe",
    ]
    mcr_candidates = [
        Path(r"C:\Program Files\MATLAB\MATLAB Compiler Runtime\v713"),
        Path(r"C:\Program Files (x86)\MATLAB\MATLAB Compiler Runtime\v713"),
    ]
    return Config(
        project_dir=project_dir,
        nnunet_predict=Path(
            r"C:\Users\andrew\anaconda3\envs\nnunet\Scripts\nnUNetv2_predict.exe"
        ),
        spm12_exe=next((path for path in spm_candidates if path.is_file()), None),
        mcr_root=next((path for path in mcr_candidates if path.is_dir()), None),
        matlab_exe="matlab",
        atlas_file=project_dir / "atlas" / "JHU_MNI_SS_WMPM_Type-II.nii",
        nnunet_raw=project_dir / "nnUNet_raw",
        nnunet_preprocessed=project_dir / "nnUNet_preprocessed",
        nnunet_results=project_dir / "nnUNet_results",
    )


def require_modules():
    missing = []
    for module in ("nibabel", "numpy", "scipy"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise RuntimeError(
            "Missing Python packages: "
            + ", ".join(missing)
            + ". Install them in the nnunet environment."
        )


def ensure_file(path: Path, label: str):
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def matlab_quote(path_or_text: Path | str) -> str:
    return str(path_or_text).replace("'", "''")


def nii_basename(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    if name.lower().endswith(".nii"):
        return name[:-4]
    return path.stem


def is_nifti_file(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _dicom_value(dataset, attr: str) -> str:
    value = getattr(dataset, attr, None)
    if value is None:
        return "None"
    text = str(value).strip()
    return text if text else "None"

def format_dicom_age(value: str) -> str:
    if value == "None":
        return value

    text = value.strip()
    if len(text) >= 2 and text[-1].isalpha():
        number = text[:-1].lstrip("0") or "0"
        unit = text[-1].upper()

        if unit == "Y":
            return number
        return f"{number}{unit}"

    return text.lstrip("0") or "0"

def format_dicom_date(value: str) -> str:
    if value == "None":
        return value
    digits = "".join(char for char in value if char.isdigit())
    if len(digits) < 8:
        return value
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"


def format_dicom_time(value: str) -> str:
    if value == "None":
        return value
    main = value.split(".", 1)[0]
    digits = "".join(char for char in main if char.isdigit())
    if len(digits) < 2:
        return value
    hours = digits[:2]
    minutes = digits[2:4] if len(digits) >= 4 else "00"
    seconds = digits[4:6] if len(digits) >= 6 else "00"
    return f"{hours}:{minutes}:{seconds}"


def read_dicom_header(dicom_files: Iterable[Path]) -> dict[str, str]:
    files = [Path(path).expanduser().resolve() for path in dicom_files]
    if not files:
        raise ValueError("No DICOM files selected.")
    ensure_file(files[0], "DICOM file")

    import pydicom

    dataset = pydicom.dcmread(str(files[0]), stop_before_pixels=True, force=True)
    study_date = format_dicom_date(_dicom_value(dataset, "StudyDate"))
    study_time = format_dicom_time(_dicom_value(dataset, "StudyTime"))
    return {
        "ID": _dicom_value(dataset, "PatientID"),
        "Name": _dicom_value(dataset, "PatientName"),
        "Gender": _dicom_value(dataset, "PatientSex"),
        "Age": format_dicom_age(_dicom_value(dataset, "PatientAge")),
        "DOB": _dicom_value(dataset, "PatientBirthDate"),
        "DOS": f"{study_date} {study_time}",
        "Weight": _dicom_value(dataset, "PatientWeight"),
    }


def safe_filename_stem(value: str) -> str:
    invalid = '<>:"/\\|?*'
    text = "".join("_" if char in invalid or ord(char) < 32 else char for char in value)
    text = text.strip(" .")
    return text or "None"


def dicom_nifti_output_from_header(dicom_folder: Path, header: dict[str, str]) -> Path:
    patient_id = safe_filename_stem(header.get("ID", "None"))
    return Path(dicom_folder).expanduser().resolve() / f"{patient_id}.nii"


def default_dicom_nifti_output(dicom_files: Iterable[Path]) -> Path:
    files = [Path(path).expanduser().resolve() for path in dicom_files]
    if not files:
        raise ValueError("No DICOM files selected.")
    header = read_dicom_header(files)
    return dicom_nifti_output_from_header(files[0].parent, header)


def dicom_files_in_folder(folder: Path) -> list[Path]:
    dicom_folder = Path(folder).expanduser().resolve()
    if not dicom_folder.is_dir():
        raise NotADirectoryError(f"DICOM folder not found: {dicom_folder}")
    return sorted(
        path for path in dicom_folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".dcm"
    )


def _copy_dicom_files_to_temp_dir(dicom_files: Iterable[Path], temp_dir: Path) -> list[Path]:
    copied_files = []
    for index, path in enumerate(dicom_files, start=1):
        source = Path(path).expanduser().resolve()
        ensure_file(source, "DICOM file")
        suffix = source.suffix or ".dcm"
        target = temp_dir / f"slice_{index:05d}{suffix}"
        shutil.copy2(source, target)
        copied_files.append(target)
    if not copied_files:
        raise ValueError("No DICOM files selected.")
    return copied_files


def _convert_dicom_with_dicom2nifti(input_dir: Path, output_file: Path) -> Path:
    import dicom2nifti

    with tempfile.TemporaryDirectory(prefix="dicom2nifti_out_") as tmp:
        output_dir = Path(tmp)
        dicom2nifti.convert_directory(
            str(input_dir),
            str(output_dir),
            compression=False,
            reorient=False,
        )
        candidates = sorted(output_dir.glob("*.nii"))
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected one converted NIfTI file, found {len(candidates)} in {output_dir}"
            )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(candidates[0]), str(output_file))
    ensure_file(output_file, "Converted NIfTI")
    return output_file


def _convert_dicom_with_simpleitk(input_dir: Path, output_file: Path) -> Path:
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(input_dir))
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in {input_dir}")
    if len(series_ids) > 1:
        raise RuntimeError(
            f"Expected one DICOM series, found {len(series_ids)}. Select one series at a time."
        )
    dicom_names = reader.GetGDCMSeriesFileNames(str(input_dir), series_ids[0])
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_file))
    ensure_file(output_file, "Converted NIfTI")
    return output_file


def convert_dicom_files_to_nifti(
    dicom_files: Iterable[Path],
    output_file: Path | None = None,
) -> Path:
    files = [Path(path).expanduser().resolve() for path in dicom_files]
    output = (
        Path(output_file).expanduser().resolve()
        if output_file is not None
        else default_dicom_nifti_output(files)
    )
    with tempfile.TemporaryDirectory(prefix="dicom_series_") as tmp:
        input_dir = Path(tmp)
        _copy_dicom_files_to_temp_dir(files, input_dir)
        try:
            return _convert_dicom_with_dicom2nifti(input_dir, output)
        except ImportError:
            return _convert_dicom_with_simpleitk(input_dir, output)


def run_command(cmd: list[str], env: dict[str, str] | None = None):
    print("Running:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}")


def spm_standalone_env(cfg: Config) -> dict[str, str]:
    env = os.environ.copy()
    if cfg.mcr_root is None:
        return env
    mcr_paths = [
        cfg.mcr_root / "runtime" / "win64",
        cfg.mcr_root / "bin" / "win64",
        cfg.mcr_root / "sys" / "os" / "win64",
        cfg.mcr_root / "sys" / "java" / "jre" / "win64" / "jre" / "bin" / "server",
    ]
    existing_paths = [str(path) for path in mcr_paths if path.exists()]
    env["PATH"] = os.pathsep.join(existing_paths + [env.get("PATH", "")])
    env.setdefault("MCR_CACHE_ROOT", str(Path(tempfile.gettempdir()) / "mcr_cache"))
    return env


def run_spm_batch(batch_file: Path, runner: str, cfg: Config):
    if runner == "standalone":
        if cfg.spm12_exe is None:
            raise FileNotFoundError("No supported SPM12 standalone executable found.")
        cmd = [str(cfg.spm12_exe), "batch", str(batch_file)]
        env = spm_standalone_env(cfg)
    elif runner == "matlab":
        matlab_cmd = (
            "spm('defaults','fmri'); "
            "spm_jobman('initcfg'); "
            f"spm_jobman('run','{matlab_quote(batch_file)}');"
        )
        cmd = [cfg.matlab_exe, "-batch", matlab_cmd]
        env = None
    else:
        raise ValueError(f"Unsupported SPM runner: {runner}")
    run_command(cmd, env=env)


def write_segment_batch(batch_file: Path, input_nii: Path):
    text = f"""
matlabbatch{{1}}.spm.tools.oldseg.data = {{'{matlab_quote(input_nii)},1'}};
matlabbatch{{1}}.spm.tools.oldseg.output.GM = [0 0 1];
matlabbatch{{1}}.spm.tools.oldseg.output.WM = [0 0 1];
matlabbatch{{1}}.spm.tools.oldseg.output.CSF = [0 0 0];
matlabbatch{{1}}.spm.tools.oldseg.output.biascor = 1;
matlabbatch{{1}}.spm.tools.oldseg.output.cleanup = 0;
matlabbatch{{1}}.spm.tools.oldseg.opts.tpm = {{
    fullfile(spm('Dir'),'toolbox','OldSeg','grey.nii')
    fullfile(spm('Dir'),'toolbox','OldSeg','white.nii')
    fullfile(spm('Dir'),'toolbox','OldSeg','csf.nii')
    }};
matlabbatch{{1}}.spm.tools.oldseg.opts.ngaus = [2
    2
    2
    4];
matlabbatch{{1}}.spm.tools.oldseg.opts.regtype = 'eastern';
matlabbatch{{1}}.spm.tools.oldseg.opts.warpreg = 1;
matlabbatch{{1}}.spm.tools.oldseg.opts.warpco = 25;
matlabbatch{{1}}.spm.tools.oldseg.opts.biasreg = 0.0001;
matlabbatch{{1}}.spm.tools.oldseg.opts.biasfwhm = 60;
matlabbatch{{1}}.spm.tools.oldseg.opts.samp = 3;
matlabbatch{{1}}.spm.tools.oldseg.opts.msk = {{''}};
"""
    batch_file.write_text(text.strip() + "\n", encoding="utf-8")


def write_normalize_batch(batch_file: Path, sn_mat: Path, resample_files: Iterable[Path]):
    resamples = "\n".join(f"    '{matlab_quote(path)},1'" for path in resample_files)
    text = f"""
matlabbatch{{1}}.spm.tools.oldnorm.write.subj.matname = {{'{matlab_quote(sn_mat)}'}};
matlabbatch{{1}}.spm.tools.oldnorm.write.subj.resample = {{
{resamples}
    }};
matlabbatch{{1}}.spm.tools.oldnorm.write.roptions.preserve = 0;
matlabbatch{{1}}.spm.tools.oldnorm.write.roptions.bb = [NaN NaN NaN
    NaN NaN NaN];
matlabbatch{{1}}.spm.tools.oldnorm.write.roptions.vox = [1 1 1];
matlabbatch{{1}}.spm.tools.oldnorm.write.roptions.interp = 1;
matlabbatch{{1}}.spm.tools.oldnorm.write.roptions.wrap = [0 0 0];
matlabbatch{{1}}.spm.tools.oldnorm.write.roptions.prefix = 'w';
"""
    batch_file.write_text(text.strip() + "\n", encoding="utf-8")


def save_reorient_mat(input_nii: Path):
    import nibabel as nib
    import numpy as np
    import scipy.io as sio

    img = nib.load(str(input_nii))
    data = np.asanyarray(img.dataobj)
    mask = data != 0
    if not mask.any():
        raise ValueError("Input image is all zeros; cannot compute foreground center.")
    coords = np.argwhere(mask)
    center_vox_spm = (coords.min(axis=0) + coords.max(axis=0)) / 2.0 + 1.0
    center_mm = img.affine @ np.r_[center_vox_spm, 1.0]
    matrix = np.eye(4)
    matrix[:3, 3] = -center_mm[:3]
    out_file = input_nii.parent / "reorient.mat"
    sio.savemat(str(out_file), {"M": matrix})
    print(f"Saved: {out_file}")


def prepare_nnunet_input(input_nii: Path, image_dir: Path) -> tuple[Path, str]:
    image_dir.mkdir(exist_ok=True)
    case_name = nii_basename(input_nii)
    nnunet_input = image_dir / f"{case_name}_0000.nii.gz"
    if input_nii.name.lower().endswith(".nii.gz"):
        shutil.copy2(input_nii, nnunet_input)
    else:
        tmp_nii = image_dir / f"{case_name}.nii"
        shutil.copy2(input_nii, tmp_nii)
        with tmp_nii.open("rb") as src, gzip.open(nnunet_input, "wb") as dst:
            shutil.copyfileobj(src, dst)
        tmp_nii.unlink()
    ensure_file(nnunet_input, "nnU-Net input")
    return nnunet_input, case_name


def run_nnunet(input_dir: Path, output_dir: Path, args, cfg: Config):
    output_dir.mkdir(exist_ok=True)
    ensure_file(cfg.nnunet_predict, "nnUNetv2_predict executable")
    env = os.environ.copy()
    env["nnUNet_raw"] = str(cfg.nnunet_raw)
    env["nnUNet_preprocessed"] = str(cfg.nnunet_preprocessed)
    env["nnUNet_results"] = str(cfg.nnunet_results)
    device = resolve_nnunet_device(args.device)
    cmd = [
        str(cfg.nnunet_predict),
        "-d",
        str(args.dataset_id),
        "-c",
        args.configuration,
        "-f",
        str(args.fold),
        "-tr",
        args.trainer,
        "-device",
        device,
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
    ]
    print(f"nnU-Net device: {device}")
    run_command(cmd, env=env)


def resolve_nnunet_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def gunzip_prediction(result_dir: Path, case_name: str) -> Path:
    gz_path = result_dir / f"{case_name}.nii.gz"
    nii_path = result_dir / f"{case_name}.nii"
    ensure_file(gz_path, "nnU-Net prediction")
    with gzip.open(gz_path, "rb") as src, nii_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    ensure_file(nii_path, "Unzipped nnU-Net prediction")
    return nii_path


def align_nifti_to_reference_space(reference_file: Path, target_file: Path) -> Path:
    import nibabel as nib
    import numpy as np

    ensure_file(reference_file, "Reference NIfTI")
    ensure_file(target_file, "Target NIfTI")
    reference_img = nib.load(str(reference_file))
    target_img = nib.load(str(target_file))
    target_data = np.asanyarray(target_img.dataobj).copy()

    if reference_img.shape[:3] != target_img.shape[:3]:
        raise ValueError(
            f"Reference shape {reference_img.shape[:3]} does not match target shape {target_img.shape[:3]}."
        )

    header = target_img.header.copy()
    aligned = nib.Nifti1Image(target_data, reference_img.affine.copy(), header)
    qform, qform_code = reference_img.get_qform(coded=True)
    sform, sform_code = reference_img.get_sform(coded=True)
    aligned.set_qform(qform if qform is not None else reference_img.affine, code=qform_code or 1)
    aligned.set_sform(sform if sform is not None else reference_img.affine, code=sform_code or 1)
    nib.save(aligned, str(target_file))
    return target_file


def find_sn_mat(input_dir: Path) -> Path:
    candidates = sorted(input_dir.glob("*seg_sn*.mat"))
    if not candidates:
        raise FileNotFoundError(f"No *seg_sn*.mat file found in {input_dir}")
    return candidates[0]


def atlas_center_mm(atlas_file: Path):
    import nibabel as nib
    import numpy as np

    atlas_img = nib.load(str(atlas_file))
    center_vox_spm = (np.array(atlas_img.shape[:3], dtype=float) + 1.0) / 2.0
    return (atlas_img.affine @ np.r_[center_vox_spm, 1.0])[:3]


def set_image_center_to_atlas(image_path: Path, atlas_center):
    import nibabel as nib
    import numpy as np

    img = nib.load(str(image_path))
    data = np.asanyarray(img.dataobj).copy()
    header = img.header.copy()
    affine = img.affine.copy()
    center_vox_spm = (np.array(img.shape[:3], dtype=float) + 1.0) / 2.0
    affine[:3, 3] = atlas_center - affine[:3, :3] @ center_vox_spm
    del img
    tmp_path = image_path.with_name(image_path.stem + "_tmp_save.nii")
    nib.save(nib.Nifti1Image(data, affine, header), str(tmp_path))
    tmp_path.replace(image_path)


def native_mask_volume_ml(native_image_file: Path, native_mask_file: Path) -> float:
    import nibabel as nib
    import numpy as np

    image = nib.load(str(native_image_file))
    mask = np.asanyarray(nib.load(str(native_mask_file)).dataobj) > 0
    zooms = image.header.get_zooms()[:3]
    voxel_volume_ml = float(np.prod(zooms)) / 1000.0
    return float(np.count_nonzero(mask) * voxel_volume_ml)


def calculate_overlap(
    atlas_file: Path,
    w_label: Path,
    native_image_file: Path | None = None,
    native_mask_file: Path | None = None,
):
    import nibabel as nib
    import numpy as np

    atlas = np.asanyarray(nib.load(str(atlas_file)).dataobj)
    roi = np.asanyarray(nib.load(str(w_label)).dataobj) > 0
    if atlas.shape != roi.shape:
        raise ValueError(f"Atlas shape {atlas.shape} does not match ROI shape {roi.shape}.")
    segment_result = np.where((atlas * roi) > 0, atlas, 0)
    stroke_total_size = np.count_nonzero(segment_result > 0)
    if stroke_total_size == 0:
        print("No atlas-overlapping lesion voxels found.")
        return []
    total_volume_ml = None
    if native_image_file is not None and native_mask_file is not None:
        total_volume_ml = native_mask_volume_ml(native_image_file, native_mask_file)
    rows = []
    for atlas_idx, name in enumerate(ATLAS_NAMES, start=1):
        stroke = np.count_nonzero(segment_result == atlas_idx)
        if stroke == 0:
            continue
        area = np.count_nonzero(atlas == atlas_idx)
        region = stroke / stroke_total_size * 100.0
        lesion = stroke / area * 100.0 if area else 0.0
        if total_volume_ml is None:
            absolute = stroke / 1000.0
        else:
            absolute = total_volume_ml * region / 100.0
        rows.append((name, region, lesion, absolute))
    return rows


def write_mask_atlas_copy(atlas_file: Path, mask_after: Path, output_file: Path) -> Path:
    import nibabel as nib
    import numpy as np

    atlas_img = nib.load(str(atlas_file))
    mask_img = nib.load(str(mask_after))
    atlas_data = np.asanyarray(atlas_img.dataobj).copy()
    mask_data = np.asanyarray(mask_img.dataobj) > 0

    if atlas_data.shape != mask_data.shape:
        raise ValueError(
            f"Atlas shape {atlas_data.shape} does not match mask shape {mask_data.shape}."
        )

    lesion_atlas = np.where(mask_data, atlas_data, 0).astype(atlas_data.dtype, copy=False)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(lesion_atlas, mask_img.affine, mask_img.header.copy())
    out_img.set_data_dtype(atlas_data.dtype)
    nib.save(out_img, str(output_file))
    return output_file

# 20260721
