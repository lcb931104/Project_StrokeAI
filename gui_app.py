from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import threading
from pathlib import Path

from pipeline_common import (
    ATLAS_NAMES,
    convert_dicom_files_to_nifti,
    dicom_files_in_folder,
    dicom_nifti_output_from_header,
    get_config,
    nii_basename,
    read_dicom_header,
)
from pipeline_runner import PipelineOptions, run_pipeline


def project_logo_path() -> Path:
    return Path(__file__).resolve().parent / "LOGO" / "LOGO.png"


def three_view_layout(width: int, height: int) -> list[tuple[str, int, int, int, int]]:
    outer_pad = 12
    gap = 12
    cell_w = max((width - outer_pad * 2 - gap) // 2, 1)
    cell_h = max((height - outer_pad * 2 - gap) // 2, 1)
    left_x = outer_pad + cell_w // 2
    right_x = outer_pad + cell_w + gap + cell_w // 2
    top_y = outer_pad + cell_h // 2
    bottom_y = outer_pad + cell_h + gap + cell_h // 2
    return [
        ("coronal", left_x, top_y, cell_w, cell_h),
        ("sagittal", right_x, top_y, cell_w, cell_h),
        ("axial", left_x, bottom_y, cell_w, cell_h),
    ]


def run_gui(options: PipelineOptions) -> int:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    import nibabel as nib
    import numpy as np
    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title("梗塞定位定量分析")
    root.geometry("1280x820")

    event_queue: queue.Queue[tuple[str, object]] = queue.Queue()
    selected_file = tk.StringVar()
    selected_input_kind = tk.StringVar(value="NIfTI")
    status = tk.StringVar(value="請選擇檔案(.nii)")
    view_mode = tk.StringVar(value="axial")
    canvas_message = tk.StringVar(value="Please choose a file\nNifTI file or DICOM folder")
    overlay_enabled = tk.BooleanVar(value=False)
    transparency = tk.DoubleVar(value=90)
    slice_index = tk.IntVar(value=0)
    ax_slice_index = tk.IntVar(value=0)
    sag_slice_index = tk.IntVar(value=0)
    cor_slice_index = tk.IntVar(value=0)
    current_images: dict[str, tuple[Path, np.ndarray]] = {}
    photo_refs: list[ImageTk.PhotoImage] = []
    render_after_id: str | None = None
    legend_window: tk.Toplevel | None = None

    display_names = ("DWI_after", "mask_atlas", "atlas")
    display_labels = {"DWI_after": "DWI", "mask_atlas": "ROI", "atlas": "Atlas"}

    def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros(arr.shape, dtype=np.uint8)
        lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            return np.zeros(arr.shape, dtype=np.uint8)
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        return (arr * 255).astype(np.uint8)

    def label_to_rgb(image: np.ndarray) -> np.ndarray:
        labels = np.asarray(image, dtype=np.int64)
        rgb = np.zeros(labels.shape + (3,), dtype=np.uint8)
        mask = labels > 0
        values = labels[mask]
        rgb[..., 0][mask] = (values * 53 + 41) % 255
        rgb[..., 1][mask] = (values * 97 + 89) % 255
        rgb[..., 2][mask] = (values * 193 + 17) % 255
        return rgb

    def get_slice(volume: np.ndarray, axis: str, index: int) -> np.ndarray:
        axis_index = {"sagittal": 0, "coronal": 1, "axial": 2}[axis]
        max_index = max(volume.shape[axis_index] - 1, 0)
        idx = min(max(index, 0), max_index)
        if axis == "sagittal":
            image = volume[idx, :, :]
        elif axis == "coronal":
            image = volume[:, idx, :]
        else:
            image = volume[:, :, idx]
        return np.rot90(image)

    def make_rgb(name: str, axis: str, index: int) -> np.ndarray:
        image = get_slice(current_images[name][1], axis, index)
        if name == "DWI_after":
            gray = normalize_to_uint8(image)
            rgb = np.repeat(gray[..., None], 3, axis=2)
            if overlay_enabled.get() and "mask_after" in current_images:
                mask = get_slice(current_images["mask_after"][1], axis, index) > 0
                alpha = max(0, min(float(transparency.get()), 100)) / 100.0
                rgb[mask, 0] = ((1.0 - alpha) * rgb[mask, 0] + alpha * 255).astype(np.uint8)
                rgb[mask, 1] = ((1.0 - alpha) * rgb[mask, 1]).astype(np.uint8)
                rgb[mask, 2] = ((1.0 - alpha) * rgb[mask, 2]).astype(np.uint8)
            return rgb
        return label_to_rgb(image)

    def fit_rgb(rgb: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
        height, width = rgb.shape[:2]
        if height == 0 or width == 0:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        scale = min(max_width / width, max_height / height, 3.0)
        out_w = max(1, int(width * scale))
        out_h = max(1, int(height * scale))
        y_idx = np.linspace(0, height - 1, out_h).astype(int)
        x_idx = np.linspace(0, width - 1, out_w).astype(int)
        return rgb[np.ix_(y_idx, x_idx)]

    def photo_from_rgb(rgb: np.ndarray) -> ImageTk.PhotoImage:
        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
        return ImageTk.PhotoImage(image)

    def set_placeholder(text: str):
        nonlocal photo_refs
        image_canvas.delete("all")
        photo_refs = []
        image_canvas.create_text(
            max(image_canvas.winfo_width() // 2, 420),
            max(image_canvas.winfo_height() // 2, 200),
            text=text,
            fill="#123040",
            font=("Microsoft JhengHei UI", 22),
            anchor="center",
            justify="center",
            width=max(image_canvas.winfo_width() - 80, 760),
        )

    def max_slice_for_mode() -> int:
        if "DWI_after" not in current_images:
            return 0
        shape = current_images["DWI_after"][1].shape[:3]
        if view_mode.get() == "sagittal":
            return max(shape[0] - 1, 0)
        if view_mode.get() == "coronal":
            return max(shape[1] - 1, 0)
        return max(shape[2] - 1, 0)
    
    def max_slice_for_axis(axis: str) -> int:
        if "DWI_after" not in current_images:
            return 0
        shape = current_images["DWI_after"][1].shape[:3]
        axis_index = {"sagittal": 0, "coronal": 1, "axial": 2}[axis]
        return max(shape[axis_index] - 1, 0)

    def configure_slice_controls():
        max_slice = max_slice_for_mode()
        slice_slider.configure(to=max_slice)
        slice_spin.configure(to=max_slice)
        if slice_index.get() > max_slice:
            slice_index.set(max_slice)
        axis_controls = (
            ("axial", ax_slice_slider, ax_slice_spin, ax_slice_index),
            ("sagittal", sag_slice_slider, sag_slice_spin, sag_slice_index),
            ("coronal", cor_slice_slider, cor_slice_spin, cor_slice_index),
        )
        for axis, slider, spin, variable in axis_controls:
            axis_max = max_slice_for_axis(axis)
            slider.configure(to=axis_max)
            spin.configure(to=axis_max)
            if variable.get() > axis_max:
                variable.set(axis_max)

    def current_slice() -> int:
        return int(slice_index.get())

    def schedule_render(*_args):
        nonlocal render_after_id
        if render_after_id is not None:
            root.after_cancel(render_after_id)
        render_after_id = root.after(40, render_images)

    def set_slice_from_slider(value: str):
        slice_index.set(int(round(float(value))))
        schedule_render()

    def set_axis_slice_from_slider(variable: tk.IntVar, value: str):
        variable.set(int(round(float(value))))
        schedule_render()

    def draw_image(name: str, axis: str, x: int, y: int, max_w: int, max_h: int, index: int | None = None):
        rgb = fit_rgb(make_rgb(name, axis, current_slice() if index is None else index), max_w, max_h)
        photo = photo_from_rgb(rgb)
        photo_refs.append(photo)
        image_canvas.create_image(x, y, image=photo, anchor="center")

    def render_single_view(canvas_width: int, canvas_height: int):
        group_w = canvas_width // 3
        max_w = max(group_w - 32, 280)
        max_h = max(canvas_height - 92, 280)
        y = canvas_height // 2 + 32
        for i, name in enumerate(display_names):
            if name not in current_images:
                continue
            x = group_w * i + group_w // 2
            image_canvas.create_text(
                x,
                34,
                text=display_labels[name],
                fill="#123040",
                font=("Microsoft JhengHei UI", 18),
                anchor="center",
            )
            draw_image(name, view_mode.get(), x, y, max_w, max_h)

    def render_three_view(canvas_width: int, canvas_height: int):
        group_w = canvas_width // 3
        label_h = 58
        view_h = max(canvas_height - label_h, 1)
        slice_by_axis = {
            "axial": int(ax_slice_index.get()),
            "sagittal": int(sag_slice_index.get()),
            "coronal": int(cor_slice_index.get()),
        }
        for i, name in enumerate(display_names):
            if name not in current_images:
                continue
            left = group_w * i
            center_x = left + group_w // 2
            image_canvas.create_text(
                center_x,
                34,
                text=display_labels[name],
                fill="#123040",
                font=("Microsoft JhengHei UI", 18),
                anchor="center",
            )
            for axis, x, y, max_w, max_h in three_view_layout(group_w, view_h):
                draw_image(name, axis, left + x, label_h + y, max_w, max_h, slice_by_axis[axis])

    def render_images():
        nonlocal render_after_id, photo_refs
        render_after_id = None
        if not current_images:
            set_placeholder(canvas_message.get())
            return
        configure_slice_controls()
        image_canvas.delete("all")
        photo_refs = []
        canvas_width = max(image_canvas.winfo_width(), 900)
        canvas_height = max(image_canvas.winfo_height(), 410)
        if view_mode.get() == "three":
            render_three_view(canvas_width, canvas_height)
        else:
            render_single_view(canvas_width, canvas_height)

    def load_nii(path: Path) -> np.ndarray:
        img = nib.load(str(path))
        data = np.asanyarray(img.dataobj)
        if data.ndim > 3:
            data = data[..., 0]
        return data.astype(np.float32, copy=False)

    def final_output_paths(input_path: Path) -> dict[str, Path]:
        case_name = nii_basename(input_path)
        patient_id = input_path.parent.name
        folder = input_path.parent
        return {
            "DWI_after": folder / f"{case_name}_after.nii",
            "mask_after": folder / f"{patient_id}_mask_after.nii",
            "mask_atlas": folder / f"{patient_id}_mask_atlas.nii",
            "圖譜": get_config().atlas_file,
        }

    def load_final_images(input_path: Path):
        paths = final_output_paths(input_path)
        missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError("找不到結果影像：\n" + "\n".join(missing))
        current_images.clear()
        canvas_message.set("Analyzing...")
        for name, path in paths.items():
            current_images[name] = (path, load_nii(path))
        if "atlas" not in current_images:
            for name in list(current_images):
                if name not in {"DWI_after", "mask_after", "mask_atlas"}:
                    current_images["atlas"] = current_images[name]
                    break
        slice_index.set(max_slice_for_mode() // 2)
        ax_slice_index.set(max_slice_for_axis("axial") // 2)
        sag_slice_index.set(max_slice_for_axis("sagittal") // 2)
        cor_slice_index.set(max_slice_for_axis("coronal") // 2)
        render_images()

    def result_text(results: list[dict]) -> str:
        if not results:
            return "沒有符合條件的定位定量結果。"
        lines = []
        for row in results:
            lines.append(
                f"{row['structure']}  Distribution: {row['region']:.4f}%  "
                f"Involvement: {row['lesion']:.4f}%  Volume: {row['volume_ml']:.4f} mL"
            )
        return "\n".join(lines)

    def clinical_output_for_structures(structures: list[str]) -> str:
        if not structures:
            return "No quantified regions."
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent / "6_clinical_symptom.py")],
            input="\n".join(structures) + "\n",
            text=True,
            capture_output=True,
        )
        if proc.returncode:
            return proc.stderr.strip() or f"Clinical symptom lookup failed: {proc.returncode}"
        return proc.stdout.strip()

    current_clinical_output = {"text": ""}
    selected_quant_widgets = {"widgets": []}
    NORMAL_ROW_BG = "white"
    HOVER_ROW_BG = "#eef6fb"
    PRESSED_ROW_BG = "#d4e6f2"
    SELECTED_ROW_BG = "#ddeef8"
    METRIC_TOOLTIPS = {
        "Distribution": "This structure's share of the total lesion volume.",
        "Involvement": "The percentage of this structure affected by the lesion.",
        "Volume": "Lesion volume within this structure, in mL.",
    }

    class ToolTip:
        def __init__(self, widget: tk.Widget, text: str):
            self.widget = widget
            self.text = text
            self.tip_window: tk.Toplevel | None = None
            widget.bind("<Enter>", self.show, add="+")
            widget.bind("<Leave>", self.hide, add="+")

        def show(self, _event=None):
            if self.tip_window is not None:
                return
            x = self.widget.winfo_rootx() + 18
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
            self.tip_window = tk.Toplevel(self.widget)
            self.tip_window.wm_overrideredirect(True)
            self.tip_window.wm_geometry(f"+{x}+{y}")
            tk.Label(
                self.tip_window,
                text=self.text,
                bg="#ffffe0",
                fg="#123040",
                relief=tk.SOLID,
                borderwidth=1,
                font=("Microsoft JhengHei UI", 10),
                padx=6,
                pady=4,
            ).pack()

        def hide(self, _event=None):
            if self.tip_window is not None:
                self.tip_window.destroy()
                self.tip_window = None

    def set_widget_background(widget: tk.Widget, color: str):
        try:
            widget.configure(bg=color)
        except tk.TclError:
            pass

    def set_row_background(widgets: tuple[tk.Widget, ...], color: str):
        for widget in widgets:
            set_widget_background(widget, color)

    def show_all_symptoms():
        diagnosis_text.delete("1.0", tk.END)
        diagnosis_text.insert(tk.END, current_clinical_output["text"])

    def show_structure_symptoms(structure: str):
        diagnosis_text.delete("1.0", tk.END)
        diagnosis_text.insert(tk.END, clinical_output_for_structures([structure]))

    def bind_clickable_quant_row(widgets: tuple[tk.Widget, ...], command):
        def is_selected() -> bool:
            return selected_quant_widgets["widgets"] == list(widgets)

        def on_enter(_event):
            if not is_selected():
                set_row_background(widgets, HOVER_ROW_BG)

        def on_leave(_event):
            if not is_selected():
                set_row_background(widgets, NORMAL_ROW_BG)

        def on_press(_event):
            set_row_background(widgets, PRESSED_ROW_BG)

        def on_release(_event):
            if selected_quant_widgets["widgets"]:
                set_row_background(tuple(selected_quant_widgets["widgets"]), NORMAL_ROW_BG)
            selected_quant_widgets["widgets"] = list(widgets)
            set_row_background(widgets, SELECTED_ROW_BG)
            command()

        for widget in widgets:
            widget.configure(cursor="hand2")
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)
            widget.bind("<ButtonPress-1>", on_press)
            widget.bind("<ButtonRelease-1>", on_release)

    def add_tooltip(widget: tk.Widget, text: str):
        tooltip = ToolTip(widget, text)
        widget.tooltip = tooltip

    def atlas_rgb(index: int) -> tuple[int, int, int]:
        return (
            (index * 53 + 41) % 255,
            (index * 97 + 89) % 255,
            (index * 193 + 17) % 255,
        )

    def atlas_index_for_name(structure_name: str) -> int | None:
        clean_name = structure_name.strip()
        if clean_name and clean_name.split(maxsplit=1)[0].isdigit():
            clean_name = clean_name.split(maxsplit=1)[1]
        try:
            return ATLAS_NAMES.index(clean_name) + 1
        except ValueError:
            return None

    def add_legend_row(parent: tk.Frame, row_index: int, name: str):
        atlas_index = atlas_index_for_name(name)
        if atlas_index is None:
            color = "#ffffff"
        else:
            r, g, b = atlas_rgb(atlas_index)
            color = f"#{r:02x}{g:02x}{b:02x}"
        swatch = tk.Canvas(parent, width=14, height=14, bg="white", highlightthickness=1, highlightbackground="#777")
        swatch.create_rectangle(1, 1, 13, 13, fill=color, outline=color)
        swatch.grid(row=row_index, column=0, padx=(4, 6), pady=2, sticky="w")
        tk.Label(parent, text=name, font=("Microsoft JhengHei UI", 10), bg="white").grid(
            row=row_index, column=1, sticky="w", pady=1
        )

    def add_quant_table_row(parent: tk.Frame, row_index: int, row: dict):
        structure = row["structure"]
        atlas_index = atlas_index_for_name(structure)
        if atlas_index is None:
            color = "#ffffff"
        else:
            r, g, b = atlas_rgb(atlas_index)
            color = f"#{r:02x}{g:02x}{b:02x}"
        swatch = tk.Canvas(parent, width=14, height=14, bg="white", highlightthickness=1, highlightbackground="#777", cursor="hand2")
        swatch.create_rectangle(1, 1, 13, 13, fill=color, outline=color)
        swatch.grid(row=row_index, column=0, padx=(4, 6), pady=2, sticky="w")
        structure_label = tk.Label(parent, text=structure, font=("Microsoft JhengHei UI", 10), bg="white", cursor="hand2")
        structure_label.grid(
            row=row_index, column=1, sticky="w", pady=1
        )
        metric_labels = (
            tk.Label(
                parent,
                text=f"Distribution: {row['region']:.2f}%",
                font=("Microsoft JhengHei UI", 10),
                bg="white",
                cursor="hand2",
            ),
            tk.Label(
                parent,
                text=f"Involvement: {row['lesion']:.2f}%",
                font=("Microsoft JhengHei UI", 10),
                bg="white",
                cursor="hand2",
            ),
            tk.Label(
                parent,
                text=f"Volume: {row['volume_ml']:.2f} mL",
                font=("Microsoft JhengHei UI", 10),
                bg="white",
                cursor="hand2",
            ),
        )
        for metric_index, metric_label in enumerate(metric_labels, start=2):
            metric_label.grid(
                row=row_index,
                column=metric_index,
                sticky="w",
                padx=(12 if metric_index == 2 else 8, 0),
                pady=1,
            )
        bind_clickable_quant_row(
            (swatch, structure_label, *metric_labels),
            lambda selected=structure: show_structure_symptoms(selected),
        )
        for metric_label, metric_name in zip(metric_labels, METRIC_TOOLTIPS):
            add_tooltip(metric_label, METRIC_TOOLTIPS[metric_name])

    def add_total_volume_row(parent: tk.Frame, row_index: int, total_volume: float):
        spacer = tk.Label(parent, text="", font=("Microsoft JhengHei UI", 10), bg="white")
        spacer.grid(row=row_index, column=0, pady=(8, 2), sticky="ew")
        title_label = tk.Label(
            parent,
            text="Total infarct volume",
            font=("Microsoft JhengHei UI", 10, "bold"),
            bg="white",
        )
        title_label.grid(row=row_index, column=1, sticky="w", pady=(8, 2))
        volume_label = tk.Label(
            parent,
            text=f"{total_volume:.2f} mL",
            font=("Microsoft JhengHei UI", 10, "bold"),
            bg="white",
        )
        volume_label.grid(row=row_index, column=2, sticky="w", padx=(12, 0), pady=(8, 2))
        bind_clickable_quant_row(
            (spacer, title_label, volume_label),
            lambda selected=show_all_symptoms: selected(),
        )

    def create_quant_table(parent: ttk.Frame):
        canvas = tk.Canvas(parent, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        inner = tk.Frame(canvas, bg="white")
        inner.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        def populate(rows: list[dict]):
            for child in inner.winfo_children():
                child.destroy()
            selected_quant_widgets["widgets"] = []
            if not rows:
                tk.Label(inner, text="No quantified regions.", font=("Microsoft JhengHei UI", 10), bg="white").grid(
                    row=0, column=0, padx=6, pady=4, sticky="w"
                )
                return
            for row_index, row in enumerate(rows):
                add_quant_table_row(inner, row_index, row)
            if rows and "total_volume_ml" in rows[0]:
                total_volume = float(rows[0]["total_volume_ml"])
            else:
                total_volume = sum(float(row.get("volume_ml", 0.0)) for row in rows)
            add_total_volume_row(inner, len(rows), total_volume)

        populate([])
        return populate

    def create_atlas_legend(parent: ttk.Frame):
        canvas = tk.Canvas(parent, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        inner = tk.Frame(canvas, bg="white")
        inner.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        def populate(structure_names: list[str]):
            for child in inner.winfo_children():
                child.destroy()
            if not structure_names:
                tk.Label(inner, text="No quantified regions.", font=("Microsoft JhengHei UI", 10), bg="white").grid(
                    row=0, column=0, padx=6, pady=4, sticky="w"
                )
                return
            for row_index, name in enumerate(structure_names):
                add_legend_row(inner, row_index, name)

        populate([])
        return populate

    def open_legend_window():
        nonlocal legend_window
        if legend_window is not None and legend_window.winfo_exists():
            legend_window.lift()
            legend_window.focus_force()
            return
        window = tk.Toplevel(root)
        legend_window = window
        window.title("Legend")
        window.geometry("520x620")
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)

        def on_close():
            nonlocal legend_window
            legend_window = None
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", on_close)

        canvas = tk.Canvas(window, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(window, orient=tk.VERTICAL, command=canvas.yview)
        inner = tk.Frame(canvas, bg="white")
        inner.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        for row_index, name in enumerate(ATLAS_NAMES):
            add_legend_row(inner, row_index, name)

    def enqueue_log(message: str):
        event_queue.put(("log", message))

    def choose_nifti_file():
        path = filedialog.askopenfilename(
            title="Choose DWI NIfTI file",
            filetypes=[
                ("NIfTI files", "*.nii *.nii.gz"),
                ("All files", "*.*"),
            ],
        )
        if path:
            selected_file.set(path)
            selected_input_kind.set("NIfTI")
            set_patient_metadata_values(empty_patient_metadata())
            status.set("已選擇 NIfTI 檔案，按 Run 開始分析")
            current_images.clear()
            set_placeholder("NIfTI selected. Press Run to analyze.")

    def choose_dicom_folder():
        folder = filedialog.askdirectory(title="Choose DICOM folder")
        if not folder:
            return
        selected_file.set(folder)
        selected_input_kind.set("DICOM")
        status.set("DICOM folder selected. Press Run to read header and convert to PatientID.nii.")
        current_images.clear()
        set_placeholder("DICOM folder selected.\nPress Run to analyze.")

    def start_pipeline():
        path = selected_file.get().strip()
        if not path:
            messagebox.showwarning("No file", "請先選擇 DWI NIfTI 或 DICOM 檔案。")
            return

        run_button.configure(state="disabled")
        choose_nifti_button.configure(state="disabled")
        choose_dicom_button.configure(state="disabled")
        populate_quant_table([])
        current_clinical_output["text"] = ""
        diagnosis_text.delete("1.0", tk.END)
        canvas_message.set("Analyzing...")
        current_images.clear()
        set_placeholder("Analyzing...")
        status.set("分析中")
        set_placeholder("Analyzing...")

        set_placeholder("Analyzing...")

        def worker():
            try:
                pipeline_input = Path(path)
                if selected_input_kind.get() == "DICOM":
                    enqueue_log("[dicom] Reading DICOM header...")
                    dicom_folder = pipeline_input
                    dicom_files = dicom_files_in_folder(dicom_folder)
                    dicom_header = read_dicom_header(dicom_files)
                    event_queue.put(("dicom_header", dicom_header))
                    dicom_output = dicom_nifti_output_from_header(dicom_folder, dicom_header)
                    enqueue_log(f"[dicom] Converting selected DICOM folder to {dicom_output.name}...")
                    pipeline_input = convert_dicom_files_to_nifti(
                        dicom_files,
                        dicom_output,
                    )
                    enqueue_log(f"[dicom] Converted NIfTI: {pipeline_input}")
                results, clinical_output, work_input_path = run_pipeline(pipeline_input, options, enqueue_log)
                event_queue.put(("done", (work_input_path, results, clinical_output)))
            except Exception as exc:
                event_queue.put(("error", exc))

        threading.Thread(target=worker, daemon=True).start()

    def poll_events():
        try:
            while True:
                kind, payload = event_queue.get_nowait()
                if kind == "log":
                    status.set(str(payload)[-80:])
                elif kind == "dicom_header":
                    set_patient_metadata_values(payload)
                elif kind == "done":
                    input_path, results, clinical_output = payload
                    run_button.configure(state="normal")
                    choose_nifti_button.configure(state="normal")
                    choose_dicom_button.configure(state="normal")
                    try:
                        load_final_images(input_path)
                    except Exception as exc:
                        status.set("結果影像載入失敗")
                        set_placeholder("結果影像載入失敗")
                        populate_quant_table([])
                        diagnosis_text.delete("1.0", tk.END)
                        diagnosis_text.insert(tk.END, f"ERROR: {exc}")
                        messagebox.showerror("Failed", str(exc))
                    else:
                        populate_quant_table(results)
                        current_clinical_output["text"] = clinical_output
                        diagnosis_text.delete("1.0", tk.END)
                        diagnosis_text.insert(tk.END, clinical_output)
                        status.set("完成")
                elif kind == "error":
                    status.set("失敗")
                    set_placeholder("分析失敗")
                    run_button.configure(state="normal")
                    choose_nifti_button.configure(state="normal")
                    choose_dicom_button.configure(state="normal")
                    populate_quant_table([])
                    current_clinical_output["text"] = ""
                    diagnosis_text.delete("1.0", tk.END)
                    diagnosis_text.insert(tk.END, f"ERROR: {payload}")
                    messagebox.showerror("Failed", str(payload))
        except queue.Empty:
            pass
        root.after(100, poll_events)

    def update_slice_control_visibility():
        if view_mode.get() == "three":
            slice_controls.grid_remove()
            three_slice_controls.grid()
        else:
            three_slice_controls.grid_remove()
            slice_controls.grid()

    def select_view(mode: str):
        view_mode.set(mode)
        update_slice_control_visibility()
        render_images()

    def on_slice_change(*_args):
        schedule_render()

    root.columnconfigure(0, weight=5, uniform="main")
    root.columnconfigure(1, weight=1, uniform="main", minsize=260)
    root.rowconfigure(0, weight=9)
    root.rowconfigure(1, weight=1)

    image_panel = ttk.Frame(root, padding=(12, 12, 8, 6))
    image_panel.grid(row=0, column=0, sticky="nsew")
    image_panel.rowconfigure(1, weight=1)
    image_panel.columnconfigure(0, weight=1)

    top = ttk.Frame(image_panel)
    top.grid(row=0, column=0, sticky="ew")

    file_entry = ttk.Entry(top, textvariable=selected_file)
    file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

    choose_nifti_button = ttk.Button(top, text="NIfTI", command=choose_nifti_file)
    choose_nifti_button.pack(side=tk.LEFT, padx=(10, 0))

    choose_dicom_button = ttk.Button(top, text="DICOM", command=choose_dicom_folder)
    choose_dicom_button.pack(side=tk.LEFT, padx=(10, 0))

    run_button = ttk.Button(top, text="Run", command=start_pipeline)
    run_button.pack(side=tk.LEFT, padx=(10, 0))

    image_canvas = tk.Canvas(
        image_panel,
        width=920,
        height=520,
        bg="white",
        highlightthickness=2,
        highlightbackground="#123040",
    )
    image_canvas.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

    patient_info_frame = ttk.Frame(image_panel, padding=(8, 8, 8, 0))
    patient_info_frame.grid(row=2, column=0, sticky="ew")

    patient_metadata_fields = ("ID", "Name", "Gender", "Age" , "Weight", "DOB", "DOS")
    patient_metadata_values = {field: tk.StringVar(value="") for field in patient_metadata_fields}
    for column in range(len(patient_metadata_fields) * 2):
        patient_info_frame.columnconfigure(column, weight=1 if column % 2 else 0)

    def empty_patient_metadata() -> dict[str, str]:
        return {field: "None" for field in patient_metadata_fields}

    def set_patient_metadata_values(metadata: dict[str, str]):
        for field in patient_metadata_fields:
            patient_metadata_values[field].set(metadata.get(field, "None"))

    for index, field in enumerate(patient_metadata_fields):
        label_column = index * 2
        entry_column = label_column + 1
        patient_info_font = ("Microsoft JhengHei UI", 10)
        style = ttk.Style()
        style.configure("PatientInfo.TEntry", font=patient_info_font)
        
        ttk.Label(patient_info_frame, text=field, font=("Microsoft JhengHei UI", 10)).grid(
            row=0,
            column=label_column,
            sticky="w",
            padx=(0 if index == 0 else 6, 4),
        )
        ttk.Entry(
            patient_info_frame,
            textvariable=patient_metadata_values[field],
            width=10,
            font=patient_info_font,
        ).grid(row=0, column=entry_column, sticky="ew", padx=(0, 8))

    controls_panel = ttk.Frame(root, padding=(12, 12, 12, 6))
    controls_panel.grid(row=0, column=1, sticky="nsew")
    controls_panel.columnconfigure(0, weight=1, minsize=260)
    control_font = ("Microsoft JhengHei UI", 12)
    style = ttk.Style()
    style.configure("Overlay.TCheckbutton", font=control_font)

    logo_path = project_logo_path()
    if logo_path.is_file():
        logo_image = Image.open(logo_path)
        logo_image.thumbnail((500, 25), Image.LANCZOS)
        logo_photo = ImageTk.PhotoImage(logo_image)
        logo_label = ttk.Label(controls_panel, image=logo_photo)
        logo_label.image = logo_photo
        logo_label.grid(row=0, column=0, sticky="n", pady=(0, 8))

    view_buttons = ttk.Frame(controls_panel)
    view_buttons.grid(row=1, column=0, sticky="ew")
    ttk.Button(view_buttons, text="Ax", width=8, command=lambda: select_view("axial")).pack(side=tk.LEFT)
    ttk.Button(view_buttons, text="Sag", width=8, command=lambda: select_view("sagittal")).pack(side=tk.LEFT, padx=(5, 0))
    ttk.Button(view_buttons, text="Cor", width=8, command=lambda: select_view("coronal")).pack(side=tk.LEFT, padx=(5, 0))
    ttk.Button(view_buttons, text="3V", width=8, command=lambda: select_view("three")).pack(side=tk.LEFT, padx=(5, 0))

    slice_controls = ttk.Frame(controls_panel)
    slice_controls.grid(row=2, column=0, sticky="ew", pady=(24, 0))
    slice_controls.columnconfigure(1, weight=1)
    ttk.Label(slice_controls, text="Slice", font=("Microsoft JhengHei UI", 12)).grid(row=0, column=0, sticky="w")
    slice_slider = ttk.Scale(
        slice_controls,
        from_=0,
        to=0,
        orient=tk.HORIZONTAL,
        variable=slice_index,
        command=set_slice_from_slider,
    )
    slice_slider.grid(row=0, column=1, sticky="ew", padx=(16, 8))
    slice_spin = ttk.Spinbox(
        slice_controls,
        from_=0,
        to=0,
        width=5,
        textvariable=slice_index,
        command=schedule_render,
    )
    slice_spin.grid(row=0, column=2, sticky="e")

    three_slice_controls = ttk.Frame(controls_panel)
    three_slice_controls.grid(row=2, column=0, sticky="ew", pady=(34, 0))
    three_slice_controls.columnconfigure(1, weight=1)

    ttk.Label(three_slice_controls, text="Ax slice", font=("Microsoft JhengHei UI", 10)).grid(
        row=0, column=0, sticky="w", pady=3
    )
    ax_slice_slider = ttk.Scale(
        three_slice_controls,
        from_=0,
        to=0,
        orient=tk.HORIZONTAL,
        variable=ax_slice_index,
        command=lambda value: set_axis_slice_from_slider(ax_slice_index, value),
    )
    ax_slice_slider.grid(row=0, column=1, sticky="ew", padx=(10, 8), pady=3)
    ax_slice_spin = ttk.Spinbox(
        three_slice_controls,
        from_=0,
        to=0,
        width=5,
        textvariable=ax_slice_index,
        command=schedule_render,
    )
    ax_slice_spin.grid(row=0, column=2, sticky="e", pady=3)

    ttk.Label(three_slice_controls, text="Sag slice", font=("Microsoft JhengHei UI", 10)).grid(
        row=1, column=0, sticky="w", pady=3
    )
    sag_slice_slider = ttk.Scale(
        three_slice_controls,
        from_=0,
        to=0,
        orient=tk.HORIZONTAL,
        variable=sag_slice_index,
        command=lambda value: set_axis_slice_from_slider(sag_slice_index, value),
    )
    sag_slice_slider.grid(row=1, column=1, sticky="ew", padx=(10, 8), pady=3)
    sag_slice_spin = ttk.Spinbox(
        three_slice_controls,
        from_=0,
        to=0,
        width=5,
        textvariable=sag_slice_index,
        command=schedule_render,
    )
    sag_slice_spin.grid(row=1, column=2, sticky="e", pady=3)

    ttk.Label(three_slice_controls, text="Cor slice", font=("Microsoft JhengHei UI", 10)).grid(
        row=2, column=0, sticky="w", pady=3
    )
    cor_slice_slider = ttk.Scale(
        three_slice_controls,
        from_=0,
        to=0,
        orient=tk.HORIZONTAL,
        variable=cor_slice_index,
        command=lambda value: set_axis_slice_from_slider(cor_slice_index, value),
    )
    cor_slice_slider.grid(row=2, column=1, sticky="ew", padx=(10, 8), pady=3)
    cor_slice_spin = ttk.Spinbox(
        three_slice_controls,
        from_=0,
        to=0,
        width=5,
        textvariable=cor_slice_index,
        command=schedule_render,
    )
    cor_slice_spin.grid(row=2, column=2, sticky="e", pady=3)

    transparency_controls = ttk.Frame(controls_panel)
    transparency_controls.grid(row=3, column=0, sticky="ew", pady=(24, 0))
    ttk.Checkbutton(
        transparency_controls,
        text="Overlay",
        variable=overlay_enabled,
        command=schedule_render,
        style="Overlay.TCheckbutton",
    ).pack(side=tk.LEFT)
    ttk.Label(transparency_controls, text="Transparency", font=control_font).pack(
        side=tk.LEFT, padx=(16, 0)
    )
    ttk.Spinbox(
        transparency_controls,
        from_=0,
        to=100,
        width=5,
        textvariable=transparency,
        command=schedule_render,
    ).pack(side=tk.LEFT, padx=(8, 0))

    status_label = ttk.Label(controls_panel, textvariable=status)
    status_label.grid(row=4, column=0, sticky="ew", pady=(24, 0))

    bottom = ttk.Frame(root, padding=(12, 0, 12, 6))
    bottom.grid(row=2, column=0, columnspan=2, sticky="nsew")
    bottom.columnconfigure(0, weight=2)
    bottom.columnconfigure(1, weight=1)
    bottom.rowconfigure(1, weight=1)

    ttk.Label(bottom, text="Infarct Quantification", font=("Microsoft JhengHei UI", 18)).grid(
        row=0, column=0, sticky="w", padx=(0, 18)
    )
    atlas_title = ttk.Frame(bottom)
    atlas_title.grid(row=0, column=0, sticky="w", padx=(260, 0))
    ttk.Button(atlas_title, text="legend", command=open_legend_window).pack(
        side=tk.LEFT, padx=(10, 0)
    )
    ttk.Label(bottom, text="Clinical Symptoms", font=("Microsoft JhengHei UI", 18)).grid(
        row=0, column=1, sticky="w", padx=(12, 0)
    )

    analysis_frame = ttk.Frame(bottom)
    analysis_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 12), pady=(6, 0))
    analysis_frame.rowconfigure(0, weight=1)
    analysis_frame.columnconfigure(0, weight=1)
    populate_quant_table = create_quant_table(analysis_frame)

    diagnosis_frame = ttk.Frame(bottom)
    diagnosis_frame.grid(row=1, column=1, sticky="nsew", padx=(12, 0), pady=(6, 0))
    diagnosis_frame.rowconfigure(0, weight=1)
    diagnosis_frame.columnconfigure(0, weight=1)
    diagnosis_text = tk.Text(diagnosis_frame, wrap=tk.WORD, font=("Microsoft JhengHei UI", 10))
    diagnosis_text.grid(row=0, column=0, sticky="nsew")
    diagnosis_scroll = ttk.Scrollbar(diagnosis_frame, command=diagnosis_text.yview)
    diagnosis_scroll.grid(row=0, column=1, sticky="ns")
    diagnosis_text.configure(yscrollcommand=diagnosis_scroll.set)

    slice_index.trace_add("write", on_slice_change)
    ax_slice_index.trace_add("write", on_slice_change)
    sag_slice_index.trace_add("write", on_slice_change)
    cor_slice_index.trace_add("write", on_slice_change)
    transparency.trace_add("write", schedule_render)
    image_canvas.bind("<Configure>", schedule_render)
    update_slice_control_visibility()
    set_placeholder("請選擇檔案")

    poll_events()
    root.mainloop()
    return 0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open the MNI lesion analysis GUI.")
    parser.add_argument("--spm-runner", choices=["standalone", "matlab"], default="standalone")
    parser.add_argument("--dataset-id", default="2")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--fold", default="1")
    parser.add_argument("--trainer", default="nnUNetTrainer_300epoch")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another nnU-Net device value.")
    parser.add_argument("--sequential", action="store_true", help="Run steps 2 and 3 one after another.")
    return parser.parse_args()


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
    return run_gui(options)


if __name__ == "__main__":
    raise SystemExit(main())

# conda activate nnunet
# cd "C:\Users\andrew\Desktop\program_python"
# python .\gui_app.py
# python "C:\Users\andrew\Desktop\program_python\gui_app.py"
# Remove-Item -Recurse -Force .\__pycache__

# 20260721
