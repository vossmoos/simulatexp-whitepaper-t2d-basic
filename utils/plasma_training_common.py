"""
Standalone helpers for plasma metabolomics classifiers used by `notebooks/`.

This file intentionally lives under `notebooks/utils/` so the external-validation
notebooks can run in a notebooks-only repo without depending on the training codebase.

Training uses only synthetic CSV rows. Real `showfile_t.txt` may be parsed for
metabolite name discovery / overlap reporting — never for fitting models.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

NOTEBOOKS_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = NOTEBOOKS_ROOT.parent

# Notebooks expect synthetic CSV here (training-only; may be absent in demo-only repos).
DEFAULT_SYNTHETIC_CSV = NOTEBOOKS_ROOT / "synthetic" / "plasma_metabolomics.csv"
NOTEBOOK_SHOWFILE = NOTEBOOKS_ROOT / "real" / "showfile_t.txt"
LEGACY_SHOWFILE = REPO_ROOT / "real" / "showfile_t.txt"


def resolve_showfile_path(explicit: Path | None = None) -> Path:
    """Prefer `notebooks/real/showfile_t.txt` when present; else `real/showfile_t.txt`."""
    if explicit is not None:
        return explicit
    if NOTEBOOK_SHOWFILE.exists():
        return NOTEBOOK_SHOWFILE
    return LEGACY_SHOWFILE


# Resolved once at import (restart kernel after adding notebooks/real copy)
DEFAULT_SHOWFILE = resolve_showfile_path()

# Model artifacts live under `notebooks/models/`
MODELS_DIR = NOTEBOOKS_ROOT / "models"

METADATA_COLUMNS = ("sample_id", "group", "study_id", "fasting_state")
LABEL_COL = "group"
BINARY_LABELS = ("control", "t2d")  # lowercase after strip
FEATURE_PREPROCESSING = "sample_rank"


def plasma_feature_columns(df: pd.DataFrame) -> list[str]:
    """Metabolite / model feature columns: every column except fixed metadata."""
    return [c for c in df.columns if c not in METADATA_COLUMNS]


def validate_synthetic_plasma_schema(df: pd.DataFrame) -> None:
    """
    Ensure the training CSV has the expected metadata headers and at least one feature column.

    Current generator format: ``sample_id, group, study_id, fasting_state`` plus metabolite
    columns (often MW **RefMet**-style names, e.g. ``2-Hydroxybutyric acid``).
    """
    missing = [c for c in METADATA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Synthetic CSV missing required columns {missing}. "
            f"Expected metadata {list(METADATA_COLUMNS)} plus metabolite columns. "
            f"Found: {list(df.columns)}"
        )
    feats = plasma_feature_columns(df)
    if not feats:
        raise ValueError(
            "No metabolite columns found after metadata. "
            f"Metadata columns are {list(METADATA_COLUMNS)}; all other columns are treated as features."
        )


def normalize_plasma_group_label(raw: object) -> str:
    """Map common label variants to canonical ``control`` / ``t2d``."""
    s = str(raw).strip().lower().replace("_", " ")
    if s in ("t2d", "type 2 diabetes", "type2 diabetes", "diabetic", "diabetes mellitus type 2"):
        return "t2d"
    if s in ("control", "non-diabetic", "non diabetic", "healthy", "normal", "neg", "negative"):
        return "control"
    return s


def format_label_for_display(label: str) -> str:
    """
    Canonical model labels are lowercase (control / t2d). Use this for inference exports
    and printed reports (Control / T2D).
    """
    key = str(label).strip().lower()
    if key == "control":
        return "Control"
    if key == "t2d":
        return "T2D"
    return str(label)


def normalize_metabolite_key(name: str) -> str:
    """Stable key for matching synthetic column names to showfile Metabolite_name / RefMet."""
    s = name.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def load_synthetic_plasma_csv(path: Path | None = None) -> pd.DataFrame:
    p = path or DEFAULT_SYNTHETIC_CSV
    if not p.exists():
        raise FileNotFoundError(
            f"Synthetic CSV not found: {p}\n"
            "Place your training file at notebooks/synthetic/plasma_metabolomics.csv "
            "(or pass an explicit path to load_synthetic_plasma_csv)."
        )
    df = pd.read_csv(p)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def apply_feature_preprocessing(
    x: np.ndarray,
    *,
    feature_preprocessing: str | None = FEATURE_PREPROCESSING,
) -> np.ndarray:
    """
    Apply the shared model feature transform.

    We keep raw CSVs unchanged and transform in-memory so training and external
    validation stay in the same numeric space.
    """
    arr = np.asarray(x, dtype=float)
    if feature_preprocessing in (None, "", "none"):
        return arr
    if feature_preprocessing == "log1p_after_impute":
        return np.log1p(np.clip(arr, a_min=0.0, a_max=None))
    if feature_preprocessing == "sample_median_norm_log1p":
        clipped = np.clip(arr, a_min=0.0, a_max=None)
        row_medians = np.median(clipped, axis=1, keepdims=True)
        row_medians = np.where(row_medians > 0, row_medians, 1.0)
        normed = clipped / row_medians
        return np.log1p(normed)
    if feature_preprocessing == "sample_rank":
        return np.apply_along_axis(
            lambda row: rankdata(row, method="average"), axis=1, arr=arr
        )
    raise ValueError(f"Unsupported feature preprocessing: {feature_preprocessing!r}")


def build_xy(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> dict[str, Any]:
    """Split into train/test, then impute and transform features in training space."""
    validate_synthetic_plasma_schema(df)
    feature_cols = plasma_feature_columns(df)

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y_raw = df[LABEL_COL].map(normalize_plasma_group_label)

    invalid = ~y_raw.isin(BINARY_LABELS)
    if invalid.any():
        bad = y_raw[invalid].unique().tolist()
        raise ValueError(f"Unsupported group labels (expected {BINARY_LABELS}): {bad}")

    le = LabelEncoder()
    le.fit(list(BINARY_LABELS))
    y = le.transform(y_raw)

    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )

    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train)
    X_test_i = imputer.transform(X_test)
    X_train_p = apply_feature_preprocessing(X_train_i, feature_preprocessing=FEATURE_PREPROCESSING)
    X_test_p = apply_feature_preprocessing(X_test_i, feature_preprocessing=FEATURE_PREPROCESSING)

    return {
        "feature_names": feature_cols,
        "label_encoder": le,
        "classes_": list(le.classes_),
        "X_train": X_train_p,
        "X_test": X_test_p,
        "y_train": y_train,
        "y_test": y_test,
        "imputer": imputer,
        "feature_preprocessing": FEATURE_PREPROCESSING,
    }


def parse_showfile_metabolite_rows(showfile_path: Path | None = None) -> list[tuple[str, str]]:
    """
    Extract (Metabolite_name, RefMet_name) from MW showfile_t-style table.

    Does not load numeric matrix values into ML training — names only.
    """
    path = resolve_showfile_path(showfile_path)
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8", errors="replace")
    # Strip HTML wrapper if present
    if "<pre>" in text:
        start = text.find("<pre>") + len("<pre>")
        end = text.find("</pre>", start)
        if end != -1:
            text = text[start:end]

    rows: list[tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip("\r")
        if not line or line.startswith("Factors\t"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        if parts[0] == "Metabolite_name":
            continue
        # Skip obvious non-metabolite header fragments
        if parts[0].lower() == "metabolite_name":
            continue
        mname, refmet = parts[0].strip(), parts[1].strip()
        if mname and mname != "-":
            rows.append((mname, refmet))
    return rows


def _factor_cell_to_label(cell: str) -> str | None:
    """Map MW Factors cell (e.g. 'Health Status:diabetic | ...') to control / t2d."""
    f = cell.lower()
    if "non-diabetic" in f:
        return "control"
    if "diabetic" in f:
        return "t2d"
    return None


def load_showfile_labeled_samples(showfile_path: Path | None = None) -> list[dict[str, Any]]:
    """
    Parse a Metabolomics Workbench wide showfile into one record per sample column.

    Each record:
      - ``sample_id``: column header from the first row
      - ``label``: ``control`` or ``t2d`` from the ``Factors`` row (``Health Status:...``)
      - ``metabolites``: raw-name -> float for each metabolite row (both Metabolite_name and
        RefMet_name keys when non-empty), for alignment via ``prepare_real_sample_row``.

    Samples with an unmapped Factors cell are skipped.
    """
    path = resolve_showfile_path(showfile_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Showfile not found: {path}\n"
            "Place showfile_t.txt under notebooks/real/ or real/ (see notebooks/real/README.md)."
        )

    text = path.read_text(encoding="utf-8", errors="replace")
    if "<pre>" in text:
        start = text.find("<pre>") + len("<pre>")
        end = text.find("</pre>", start)
        if end != -1:
            text = text[start:end]

    lines = [ln.rstrip("\r\n") for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return []

    header = lines[0].split("\t")
    if not header or header[0].strip() != "Metabolite_name":
        raise ValueError(
            f"Expected first column 'Metabolite_name' in showfile header, got {header[0]!r} in {path}"
        )

    sample_ids = [h.strip() for h in header[2:]]
    n = len(sample_ids)

    factors_row = lines[1].split("\t")
    if not factors_row or factors_row[0].strip() != "Factors":
        raise ValueError("Expected second row to start with 'Factors'")

    factor_cells = factors_row[2 : 2 + n]
    if len(factor_cells) < n:
        factor_cells = factor_cells + [""] * (n - len(factor_cells))

    labels: list[str | None] = [_factor_cell_to_label(c) for c in factor_cells]

    metabolite_lines = lines[2:]
    # (mname, refmet, value strings per sample)
    rows_data: list[tuple[str, str, list[str]]] = []
    for line in metabolite_lines:
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        if parts[0].strip() == "Metabolite_name":
            continue
        mname, refmet = parts[0].strip(), parts[1].strip()
        vals = parts[2 : 2 + n]
        if len(vals) < n:
            vals = vals + [""] * (n - len(vals))
        else:
            vals = vals[:n]
        rows_data.append((mname, refmet, vals))

    out: list[dict[str, Any]] = []
    for j, sid in enumerate(sample_ids):
        lab = labels[j] if j < len(labels) else None
        if lab is None:
            continue
        meta: dict[str, float] = {}
        for mname, refmet, vals in rows_data:
            cell = vals[j].strip() if j < len(vals) else ""
            if not cell:
                continue
            try:
                v = float(cell)
            except ValueError:
                continue
            if mname and mname != "-":
                meta[mname] = v
            if refmet and refmet != "-":
                meta[refmet] = v
        out.append({"sample_id": sid, "label": lab, "metabolites": meta})

    return out


def feature_overlap_with_showfile(
    feature_names: list[str],
    showfile_rows: list[tuple[str, str]],
) -> dict[str, Any]:
    keys_feat = {normalize_metabolite_key(n): n for n in feature_names}
    show_keys: dict[str, tuple[str, str]] = {}
    for m, r in showfile_rows:
        show_keys[normalize_metabolite_key(m)] = (m, r)
        if r and r != "-":
            show_keys[normalize_metabolite_key(r)] = (m, r)

    matched: list[dict[str, str]] = []
    missing_in_showfile: list[str] = []
    for orig in feature_names:
        k = normalize_metabolite_key(orig)
        if k in show_keys:
            m, r = show_keys[k]
            matched.append(
                {
                    "synthetic_column": orig,
                    "showfile_metabolite_name": m,
                    "showfile_refmet_name": r,
                }
            )
        else:
            missing_in_showfile.append(orig)

    return {
        "n_features": len(feature_names),
        "n_showfile_metabolites": len(showfile_rows),
        "n_matched_keys": len(matched),
        "matched": matched,
        "missing_in_showfile": missing_in_showfile,
    }


def save_manifest(
    path: Path,
    *,
    feature_names: list[str],
    label_classes: list[str],
    synthetic_csv: str,
    notes: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_columns": feature_names,
        "metadata_columns": list(METADATA_COLUMNS),
        "label_column": LABEL_COL,
        "label_classes": label_classes,
        "feature_preprocessing": FEATURE_PREPROCESSING,
        "synthetic_training_csv": synthetic_csv,
        "external_validation": {
            "reference_file": "real/showfile_t.txt (preferred)",
            "format": "Metabolomics Workbench wide table: rows=metabolites, columns=samples; "
            "map Metabolite_name and RefMet_name to feature_columns via normalize_metabolite_key, "
            "then apply the saved feature_preprocessing before inference.",
            "real_data_used_in_training": False,
        },
        "notes": notes,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepare_real_sample_row(
    metabolite_name_to_value: dict[str, float],
    feature_names: list[str],
    imputer: SimpleImputer,
    *,
    feature_preprocessing: str | None = FEATURE_PREPROCESSING,
) -> np.ndarray:
    """
    Build a single feature vector for external validation.

    metabolite_name_to_value keys should be normalized with normalize_metabolite_key
    or match synthetic column names case-insensitively.
    """
    lookup = {normalize_metabolite_key(k): v for k, v in metabolite_name_to_value.items()}
    row = []
    for col in feature_names:
        k = normalize_metabolite_key(col)
        val = lookup.get(k)
        row.append(val if val is not None else np.nan)
    x = np.array([row], dtype=float)
    x_i = imputer.transform(x)
    return apply_feature_preprocessing(x_i, feature_preprocessing=feature_preprocessing)


def count_aligned_metabolite_features(
    metabolite_name_to_value: dict[str, float],
    feature_names: list[str],
) -> int:
    """
    Number of model columns that receive a real value from the showfile dict before imputation.

    If this is small relative to len(feature_names), external predictions are mostly medians from
    training — expect poor calibration vs real cohorts (often one predicted class).
    """
    lookup = {normalize_metabolite_key(k): v for k, v in metabolite_name_to_value.items()}
    n = 0
    for col in feature_names:
        k = normalize_metabolite_key(col)
        val = lookup.get(k)
        if val is None:
            continue
        if isinstance(val, float) and np.isnan(val):
            continue
        n += 1
    return n


