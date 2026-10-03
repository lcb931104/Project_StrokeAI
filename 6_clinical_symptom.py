from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def load_symptom_table(project_dir: Path) -> pd.DataFrame:
    table_dir = project_dir / "clinical_symptoms_table"
    candidates = sorted(table_dir.glob("*.xlsx"))
    if not candidates:
        raise FileNotFoundError(f"No .xlsx file found in {table_dir}")
    return pd.read_excel(candidates[0])


def normalize_structure_name(name: str) -> str:
    name = name.strip()
    if name and name.split(maxsplit=1)[0].isdigit():
        return name.split(maxsplit=1)[1].strip()
    return name


def collect_symptoms(structures: list[str], table: pd.DataFrame) -> list[str]:
    if table.shape[1] < 2:
        raise ValueError("Clinical symptom table must have at least two columns.")

    requested = {normalize_structure_name(name) for name in structures if name.strip()}
    symptoms: list[str] = []

    for _, row in table.iterrows():
        region = normalize_structure_name(str(row.iloc[0]))
        if region not in requested:
            continue
        symptom_text = "" if pd.isna(row.iloc[1]) else str(row.iloc[1])
        for symptom in symptom_text.splitlines():
            symptom = symptom.strip()
            if symptom and symptom not in symptoms:
                symptoms.append(symptom)

    return symptoms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 6: map quantified brain regions to possible clinical symptoms."
    )
    parser.add_argument("structures", nargs="*", help="Brain region names from localization results.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    structures = args.structures
    if not structures and not sys.stdin.isatty():
        structures = [line.strip() for line in sys.stdin if line.strip()]

    table = load_symptom_table(Path(__file__).resolve().parent)
    symptoms = collect_symptoms(structures, table)

    if not symptoms:
        print("No matched clinical symptoms.")
        return 0

    for symptom in symptoms:
        print(symptom)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# 20260721