"""Functions for the final retrospective biogas analysis.

Parameters and calculations follow the final manuscript. The temporal partitions
are disjoint, but data preparation and retrospective analysis involved the existing
evaluation window; this reproduction does not establish independent validation.
Run ../run.py to reconstruct inputs, fit models and calculate evaluation outputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns
from openpyxl import load_workbook
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from statsmodels.stats.multitest import multipletests

try:
    import catboost as cb
except Exception:  # pragma: no cover - optional dependency guard
    cb = None
try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None
try:
    import shap
except Exception:  # pragma: no cover
    shap = None
try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None


ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = ROOT / "tables"
FIG_DIR = ROOT / "figures"
MODEL_DIR = ROOT / "models"
AUDIT_DIR = ROOT / "audit"

TARGET = "biogas_rate_STP_mL_d"
DATE_COL = "date"
RANDOM_STATE = 42
BOOTSTRAP_REPS = 1000

SPLIT_WINDOWS = {
    "train": ("2024-04-22", "2024-08-07"),
    "validation": ("2024-08-08", "2024-09-04"),
    "calibration": ("2024-09-05", "2024-09-20"),
    "test": ("2024-09-21", "2024-10-18"),
}

COLORS = {
    # Exact visual tokens from the original manuscript figure script.
    "navy": "#3C5488",
    "blue": "#4DBBD5",
    "green": "#00A087",
    "red": "#E64B35",
    "coral": "#F39B7F",
    "teal": "#91D1C2",
    "lavender": "#8491B4",
    "brown": "#7E6148",
    "sand": "#B09C85",
    "ink": "#2B2B2B",
    "grey": "#8A8F96",
}

REACTORS = ["RI-FLEX", "R2-FLEX", "R3-FIXED DOME", "R4-FIXED DOME"]
REACTOR_LABELS = {"RI-FLEX": "R1-FLEX"}
REACTOR_COLORS = {
    "RI-FLEX": COLORS["green"],
    "R2-FLEX": COLORS["blue"],
    "R3-FIXED DOME": COLORS["coral"],
    "R4-FIXED DOME": COLORS["lavender"],
}
MODEL_COLORS = {
    "ExtraTrees": COLORS["navy"],
    "XGBoost": COLORS["blue"],
    "CatBoost": COLORS["green"],
    "RandomForest": COLORS["coral"],
    "LightGBM": COLORS["lavender"],
    "Ridge": COLORS["sand"],
    "KNN": COLORS["red"],
    "MLP": COLORS["teal"],
    "SVR": COLORS["brown"],
}
AXIS_LABEL_SIZE = 18.0
TICK_LABEL_SIZE = 14.2
LEGEND_SIZE = 13.2


def reactor_label(name: str) -> str:
    return REACTOR_LABELS.get(name, name)


def reactor_tick_labels(order: list[str]) -> list[str]:
    return [reactor_label(value).replace(" ", "\n") for value in order]


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, MODEL_DIR, AUDIT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_style() -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "axes.linewidth": 1.7,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "xtick.major.size": 7.2,
            "ytick.major.size": 7.2,
            "xtick.major.width": 1.55,
            "ytick.major.width": 1.55,
            "xtick.minor.size": 4.2,
            "ytick.minor.size": 4.2,
            "xtick.minor.width": 1.15,
            "ytick.minor.width": 1.15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax, xlabel: str = "", ylabel: str = "") -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(COLORS["ink"])
        ax.spines[spine].set_linewidth(1.7)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(axis="x", which="major", bottom=True, top=False, colors=COLORS["ink"], width=1.55, length=7.2, direction="out", labelsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="y", which="major", left=True, right=False, colors=COLORS["ink"], width=1.55, length=7.2, direction="out", labelsize=TICK_LABEL_SIZE)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(TICK_LABEL_SIZE)
    ax.set_xlabel(xlabel, fontweight="bold", labelpad=7)
    ax.set_ylabel(ylabel, fontweight="bold", labelpad=7)


def numeric_ticks(ax, x: bool = True, y: bool = True, n: int = 5) -> None:
    if x:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=n, prune=None))
    if y:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=n, prune=None))


def label_panels(axes, x: float = -0.065, y: float = 1.045) -> None:
    for label, ax in zip("abcdefghijklmnopqrstuvwxyz", np.asarray(axes).reshape(-1)):
        ax.text(x, y, f"({label})", transform=ax.transAxes, fontsize=20, fontweight="normal", va="top")


def save_composite_figure(fig: plt.Figure, stem: str, axes) -> None:
    folder = FIG_DIR / stem
    sub = folder / "subfigures"
    folder.mkdir(parents=True, exist_ok=True)
    sub.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.35)
    for suffix, target in [(".png", folder / f"{stem}.png"), (".pdf", folder / f"{stem}.pdf")]:
        fig.savefig(target, dpi=600 if suffix == ".png" else None, bbox_inches="tight", facecolor="white")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    panel_names = []
    for label, ax in zip("abcdefghijklmnopqrstuvwxyz", np.asarray(axes).reshape(-1)):
        bbox = ax.get_tightbbox(renderer).expanded(1.04, 1.08).transformed(fig.dpi_scale_trans.inverted())
        panel_names.append(label)
        for suffix, target in [(".png", sub / f"{stem}_{label}.png"), (".pdf", sub / f"{stem}_{label}.pdf")]:
            fig.savefig(target, dpi=600 if suffix == ".png" else None, bbox_inches=bbox, facecolor="white")
    expected_panel_files = {
        sub / f"{stem}_{label}{suffix}"
        for label in panel_names
        for suffix in (".png", ".pdf")
    }
    stale_panel_files = [
        path for path in sub.glob(f"{stem}_*")
        if path.suffix.lower() in {".png", ".pdf"} and path not in expected_panel_files
    ]
    if stale_panel_files:
        archive = ROOT / "qa" / "deprecated_figure_drafts" / stem / "subfigures"
        archive.mkdir(parents=True, exist_ok=True)
        for stale in stale_panel_files:
            target = archive / stale.name
            if target.exists():
                target = archive / f"{stale.stem}_{stale.stat().st_mtime_ns}{stale.suffix}"
            stale.replace(target)
    (sub / "README.md").write_text(
        "Composite figure panels exported from the same locked-analysis canvas: " + ", ".join(panel_names) + ".\n",
        encoding="utf-8",
    )
    plt.close(fig)






def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", encoder),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def model_library() -> dict[str, object]:
    models: dict[str, object] = {
        "Ridge": Ridge(alpha=10.0),
        "KNN": KNeighborsRegressor(n_neighbors=18, weights="distance"),
        "SVR": SVR(C=10.0, epsilon=50.0, gamma="scale"),
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            max_depth=7,
            min_samples_leaf=8,
            max_features=0.75,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=400,
            max_depth=7,
            min_samples_leaf=8,
            max_features=0.75,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(32, 16),
            alpha=0.01,
            learning_rate_init=0.002,
            early_stopping=True,
            max_iter=1500,
            random_state=RANDOM_STATE,
        ),
    }
    if xgb is not None:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=180,
            learning_rate=0.03,
            max_depth=2,
            min_child_weight=12,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_lambda=15,
            reg_alpha=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
    if lgb is not None:
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=180,
            learning_rate=0.03,
            max_depth=2,
            num_leaves=5,
            min_child_samples=25,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_lambda=15,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    if cb is not None:
        models["CatBoost"] = cb.CatBoostRegressor(
            iterations=180,
            learning_rate=0.03,
            depth=2,
            l2_leaf_reg=15,
            random_strength=1.5,
            random_state=RANDOM_STATE,
            verbose=0,
            allow_writing_files=False,
        )
    return models


def make_pipeline(model: object, numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor(numeric, categorical)), ("model", model)])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, mase_scale: float) -> dict[str, float]:
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE_mL": rmse,
        "MAE_mL": mae,
        "MASE": float(mae / mase_scale) if mase_scale > 0 else float("nan"),
    }


def training_mase_scale(train: pd.DataFrame) -> float:
    errors: list[float] = []
    for _, group in train.sort_values(["reactor_id", DATE_COL]).groupby("reactor_id"):
        diff = group[TARGET].diff().abs().dropna()
        errors.extend(diff.tolist())
    return float(np.mean(errors))


def rolling_cv(
    train: pd.DataFrame,
    features: list[str],
    numeric: list[str],
    categorical: list[str],
    models: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(train[DATE_COL].unique()))
    splitter = TimeSeriesSplit(n_splits=5)
    fold_rows: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []
    for model_name, estimator in models.items():
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(dates), start=1):
            fold_train = train[train[DATE_COL].isin(set(dates[train_idx]))]
            fold_valid = train[train[DATE_COL].isin(set(dates[valid_idx]))]
            pipe = make_pipeline(clone(estimator), numeric, categorical)
            pipe.fit(fold_train[features], fold_train[TARGET])
            pred = pipe.predict(fold_valid[features])
            scale = training_mase_scale(fold_train)
            metrics = regression_metrics(fold_valid[TARGET].to_numpy(), pred, scale)
            fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "train_start": fold_train[DATE_COL].min(),
                    "train_end": fold_train[DATE_COL].max(),
                    "validation_start": fold_valid[DATE_COL].min(),
                    "validation_end": fold_valid[DATE_COL].max(),
                    **metrics,
                }
            )
            for row_index, prediction in zip(fold_valid.index, pred):
                oof_rows.append(
                    {
                        "model": model_name,
                        "fold": fold,
                        "row_index": int(row_index),
                        DATE_COL: fold_valid.loc[row_index, DATE_COL],
                        "reactor_id": fold_valid.loc[row_index, "reactor_id"],
                        "observed_mL": float(fold_valid.loc[row_index, TARGET]),
                        "predicted_mL": float(prediction),
                    }
                )
    return pd.DataFrame(fold_rows), pd.DataFrame(oof_rows)


def validation_evaluation(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    numeric: list[str],
    categorical: list[str],
    models: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    scale = training_mase_scale(train)
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for model_name, estimator in models.items():
        pipe = make_pipeline(clone(estimator), numeric, categorical)
        pipe.fit(train[features], train[TARGET])
        pred = pipe.predict(validation[features])
        metric_rows.append({"model": model_name, **regression_metrics(validation[TARGET].to_numpy(), pred, scale)})
        for row_index, prediction in zip(validation.index, pred):
            prediction_rows.append(
                {
                    "model": model_name,
                    "row_index": int(row_index),
                    DATE_COL: validation.loc[row_index, DATE_COL],
                    "reactor_id": validation.loc[row_index, "reactor_id"],
                    "observed_mL": float(validation.loc[row_index, TARGET]),
                    "predicted_mL": float(prediction),
                    "absolute_error_mL": float(abs(validation.loc[row_index, TARGET] - prediction)),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    selected = metrics.sort_values(["RMSE_mL", "MASE", "MAE_mL", "model"]).iloc[0]["model"]
    return metrics, predictions, str(selected)


def holm_pairwise(validation_predictions: pd.DataFrame, selected_model: str) -> pd.DataFrame:
    pivot = (
        validation_predictions.groupby(["date", "model"], as_index=False)["absolute_error_mL"]
        .mean()
        .pivot(index="date", columns="model", values="absolute_error_mL")
    )
    rows: list[dict[str, object]] = []
    for model in pivot.columns:
        if model == selected_model:
            continue
        paired = pivot[[selected_model, model]].dropna()
        if len(paired) < 5 or np.allclose(paired[selected_model], paired[model]):
            p_value = 1.0
            statistic = 0.0
        else:
            statistic, p_value = stats.wilcoxon(
                paired[selected_model], paired[model], alternative="two-sided", zero_method="wilcox"
            )
        rows.append(
            {
                "reference_model": selected_model,
                "comparison_model": model,
                "n_dates": len(paired),
                "wilcoxon_statistic": float(statistic),
                "p_raw": float(p_value),
                "median_daily_abs_error_difference_mL": float(
                    np.median(paired[model] - paired[selected_model])
                ),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        rejected, corrected, _, _ = multipletests(result["p_raw"], method="holm")
        result["p_holm"] = corrected
        result["reject_holm_0_05"] = rejected
    return result


def bootstrap_metric_intervals(
    frame: pd.DataFrame,
    observed_col: str,
    predicted_col: str,
    mase_scale: float,
    reps: int = BOOTSTRAP_REPS,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    dates = np.array(sorted(frame[DATE_COL].unique()))
    estimates: list[dict[str, float]] = []
    for _ in range(reps):
        sampled_dates = rng.choice(dates, size=len(dates), replace=True)
        pieces = [frame.loc[frame[DATE_COL] == date] for date in sampled_dates]
        sample = pd.concat(pieces, ignore_index=True)
        if sample[observed_col].nunique() < 2:
            continue
        estimates.append(
            regression_metrics(
                sample[observed_col].to_numpy(), sample[predicted_col].to_numpy(), mase_scale
            )
        )
    boot = pd.DataFrame(estimates)
    point = regression_metrics(frame[observed_col].to_numpy(), frame[predicted_col].to_numpy(), mase_scale)
    rows = []
    for metric, value in point.items():
        rows.append(
            {
                "metric": metric,
                "estimate": value,
                "ci95_low": float(boot[metric].quantile(0.025)),
                "ci95_high": float(boot[metric].quantile(0.975)),
                "bootstrap_unit": "date block",
                "bootstrap_reps": len(boot),
            }
        )
    return pd.DataFrame(rows)


def final_fit_and_test(
    eligible: pd.DataFrame,
    features: list[str],
    numeric: list[str],
    categorical: list[str],
    models: dict[str, object],
    selected_model: str,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    train = eligible.loc[eligible["split"] == "train"].copy()
    validation = eligible.loc[eligible["split"] == "validation"].copy()
    calibration = eligible.loc[eligible["split"] == "calibration"].copy()
    test = eligible.loc[eligible["split"] == "test"].copy()
    development = pd.concat([train, validation], ignore_index=True)

    final_pipe = make_pipeline(clone(models[selected_model]), numeric, categorical)
    final_pipe.fit(development[features], development[TARGET])
    calibration_pred = final_pipe.predict(calibration[features])
    absolute_residual = np.abs(calibration[TARGET].to_numpy() - calibration_pred)
    q90 = float(np.quantile(absolute_residual, 0.90, method="higher"))
    q95 = float(np.quantile(absolute_residual, 0.95, method="higher"))

    test_pred = final_pipe.predict(test[features])
    prediction = test[[DATE_COL, "reactor_id", "reactor_type", TARGET, "days_since_prev"]].copy()
    prediction = prediction.rename(columns={TARGET: "observed_mL"})
    prediction["predicted_mL"] = test_pred
    prediction["lower90_mL"] = test_pred - q90
    prediction["upper90_mL"] = test_pred + q90
    prediction["lower95_mL"] = test_pred - q95
    prediction["upper95_mL"] = test_pred + q95
    prediction["covered90"] = prediction["observed_mL"].between(prediction["lower90_mL"], prediction["upper90_mL"])
    prediction["covered95"] = prediction["observed_mL"].between(prediction["lower95_mL"], prediction["upper95_mL"])

    scale = training_mase_scale(development)
    overall = regression_metrics(prediction["observed_mL"].to_numpy(), prediction["predicted_mL"].to_numpy(), scale)

    reactor_rows = []
    for reactor, subset in prediction.groupby("reactor_id"):
        row = {
            "reactor_id": reactor,
            "n_records": len(subset),
            "n_dates": subset[DATE_COL].nunique(),
            **regression_metrics(subset["observed_mL"].to_numpy(), subset["predicted_mL"].to_numpy(), scale),
            "coverage90": float(subset["covered90"].mean()),
            "coverage95": float(subset["covered95"].mean()),
            "mean_interval_width90_mL": 2 * q90,
            "mean_interval_width95_mL": 2 * q95,
        }
        reactor_rows.append(row)
    reactor_metrics = pd.DataFrame(reactor_rows)

    coverage = pd.DataFrame(
        [
            {
                "group": "overall",
                "n_records": len(prediction),
                "coverage90": float(prediction["covered90"].mean()),
                "coverage95": float(prediction["covered95"].mean()),
                "width90_mL": 2 * q90,
                "width95_mL": 2 * q95,
            }
        ]
        + [
            {
                "group": reactor,
                "n_records": len(subset),
                "coverage90": float(subset["covered90"].mean()),
                "coverage95": float(subset["covered95"].mean()),
                "width90_mL": 2 * q90,
                "width95_mL": 2 * q95,
            }
            for reactor, subset in prediction.groupby("reactor_id")
        ]
    )
    quantiles = {"q90_mL": q90, "q95_mL": q95, "calibration_n": int(len(calibration))}
    return final_pipe, prediction, reactor_metrics, coverage, {**overall, **quantiles, "mase_scale_mL": scale}


def baseline_table(eligible: pd.DataFrame, predictions: pd.DataFrame, mase_scale: float) -> pd.DataFrame:
    test = eligible.loc[eligible["split"] == "test"].copy()
    rows = []
    for name, column in [("Persistence", "biogas_lag1"), ("Rolling-3", "biogas_roll3"), ("Rolling-7", "biogas_roll7")]:
        subset = test.loc[test[column].notna()].copy()
        metrics = regression_metrics(subset[TARGET].to_numpy(), subset[column].to_numpy(), mase_scale)
        rows.append({"model": name, "n_records": len(subset), **metrics})
    model_metrics = regression_metrics(
        predictions["observed_mL"].to_numpy(), predictions["predicted_mL"].to_numpy(), mase_scale
    )
    persistence_rmse = next(row["RMSE_mL"] for row in rows if row["model"] == "Persistence")
    rows.append(
        {
            "model": "Selected ML model",
            "n_records": len(predictions),
            **model_metrics,
            "RMSE_skill_vs_persistence": 1 - model_metrics["RMSE_mL"] / persistence_rmse,
        }
    )
    table = pd.DataFrame(rows)
    if "RMSE_skill_vs_persistence" not in table.columns:
        table["RMSE_skill_vs_persistence"] = np.nan
    return table


def peak_gap_diagnostics(eligible: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    development = eligible.loc[eligible["split"].isin(["train", "validation"])]
    peak_threshold = float(development[TARGET].quantile(0.90))
    pred = predictions.copy()
    pred["is_peak"] = pred["observed_mL"] >= peak_threshold
    peak_rows = []
    for label, subset in [("all test", pred), ("test peaks", pred.loc[pred["is_peak"]])]:
        peak_rows.append(
            {
                "subset": label,
                "n_records": len(subset),
                "threshold_mL": peak_threshold,
                "MAE_mL": float(mean_absolute_error(subset["observed_mL"], subset["predicted_mL"])) if len(subset) else np.nan,
                "mean_bias_mL": float((subset["predicted_mL"] - subset["observed_mL"]).mean()) if len(subset) else np.nan,
            }
        )
    gap_rows = []
    for group_name, subset in [("consecutive observations", pred.loc[pred["days_since_prev"] <= 1]), ("after gaps", pred.loc[pred["days_since_prev"] > 1])]:
        gap_rows.append(
            {
                "subset": group_name,
                "n_records": len(subset),
                "MAE_mL": float(mean_absolute_error(subset["observed_mL"], subset["predicted_mL"])) if len(subset) else np.nan,
                "RMSE_mL": float(math.sqrt(mean_squared_error(subset["observed_mL"], subset["predicted_mL"]))) if len(subset) else np.nan,
            }
        )
    return pd.DataFrame(peak_rows), pd.DataFrame(gap_rows)


def sensitivity_table(
    eligible: pd.DataFrame,
    features: list[str],
    numeric: list[str],
    categorical: list[str],
    selected_model: str,
    models: dict[str, object],
) -> pd.DataFrame:
    train = eligible.loc[eligible["split"] == "train"].copy()
    validation = eligible.loc[eligible["split"] == "validation"].copy()
    calibration = eligible.loc[eligible["split"] == "calibration"].copy()
    test = eligible.loc[eligible["split"] == "test"].copy()
    development = pd.concat([train, validation], ignore_index=True)
    # Keep calibration outcomes out of model fitting in every scenario.
    full_train = development.copy()
    scale = training_mase_scale(development)

    q1, q3 = eligible[TARGET].quantile([0.25, 0.75])
    upper = float(q3 + 3 * (q3 - q1))
    scenarios = {
        "Primary": (full_train, test, features, numeric),
        "No weather": (
            full_train,
            test,
            [c for c in features if not c.startswith("daily_") and c != "air_temp_in_situ_lag1"],
            [c for c in numeric if not c.startswith("daily_") and c != "air_temp_in_situ_lag1"],
        ),
        "Parsimonious history + operation": (
            full_train,
            test,
            [c for c in ["biogas_lag1", "biogas_roll3", "days_since_prev", "manure_fed_kg", "water_kg"] if c in features]
            + categorical,
            [c for c in ["biogas_lag1", "biogas_roll3", "days_since_prev", "manure_fed_kg", "water_kg"] if c in numeric],
        ),
        "Consecutive only": (full_train.loc[full_train["days_since_prev"] <= 1], test.loc[test["days_since_prev"] <= 1], features, numeric),
        "Exclude R4": (full_train.loc[full_train["reactor_id"] != "R4-FIXED DOME"], test.loc[test["reactor_id"] != "R4-FIXED DOME"], features, numeric),
        "3xIQR sensitivity": (full_train.loc[full_train[TARGET] <= upper], test.loc[test[TARGET] <= upper], features, numeric),
    }
    rows = []
    for scenario, (fit_frame, test_frame, scenario_features, scenario_numeric) in scenarios.items():
        pipe = make_pipeline(clone(models[selected_model]), scenario_numeric, categorical)
        pipe.fit(fit_frame[scenario_features], fit_frame[TARGET])
        pred = pipe.predict(test_frame[scenario_features])
        rows.append(
            {
                "scenario": scenario,
                "train_n": len(fit_frame),
                "test_n": len(test_frame),
                **regression_metrics(test_frame[TARGET].to_numpy(), pred, scale),
            }
        )
    return pd.DataFrame(rows)


def unit_safe_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Make daily-rate units explicit in machine-readable outputs."""

    return frame.rename(
        columns={
            "RMSE_mL": "RMSE_mL_d",
            "MAE_mL": "MAE_mL_d",
            "q90_mL": "q90_mL_d",
            "q95_mL": "q95_mL_d",
            "mase_scale_mL": "mase_scale_mL_d",
            "observed_mL": "observed_mL_d",
            "predicted_mL": "predicted_mL_d",
            "lower90_mL": "lower90_mL_d",
            "upper90_mL": "upper90_mL_d",
            "lower95_mL": "lower95_mL_d",
            "upper95_mL": "upper95_mL_d",
            "absolute_error_mL": "absolute_error_mL_d",
            "mean_interval_width90_mL": "mean_interval_width90_mL_d",
            "mean_interval_width95_mL": "mean_interval_width95_mL_d",
            "width90_mL": "width90_mL_d",
            "width95_mL": "width95_mL_d",
            "threshold_mL": "threshold_mL_d",
            "mean_bias_mL": "mean_bias_mL_d",
            "permutation_importance_RMSE_increase_mL": "permutation_importance_RMSE_increase_mL_d",
            "importance_sd_mL": "importance_sd_mL_d",
            "median_daily_abs_error_difference_mL": "median_daily_abs_error_difference_mL_d",
        }
    )


def interpretation_outputs(
    final_pipe: Pipeline,
    eligible: pd.DataFrame,
    predictions: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    test = eligible.loc[eligible["split"] == "test"].copy()
    result = permutation_importance(
        final_pipe,
        test[features],
        test[TARGET],
        scoring="neg_root_mean_squared_error",
        n_repeats=30,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": features,
            "permutation_importance_RMSE_increase_mL": result.importances_mean,
            "importance_sd_mL": result.importances_std,
        }
    ).sort_values("permutation_importance_RMSE_increase_mL", ascending=False)

    # SHAP is computed only for the already-frozen estimator and transformed test matrix.
    model = final_pipe.named_steps["model"]
    pre = final_pipe.named_steps["preprocessor"]
    if shap is not None and model.__class__.__name__ in {
        "ExtraTreesRegressor",
        "RandomForestRegressor",
        "XGBRegressor",
        "LGBMRegressor",
        "CatBoostRegressor",
    }:
        transformed = pre.transform(test[features])
        names = pre.get_feature_names_out()
        try:
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(transformed)
            mean_abs = np.abs(np.asarray(values)).mean(axis=0)
            pd.DataFrame({"transformed_feature": names, "mean_abs_SHAP_mL_d": mean_abs}).sort_values(
                "mean_abs_SHAP_mL_d", ascending=False
            ).to_csv(TABLE_DIR / "Table_S13_frozen_model_SHAP.csv", index=False, encoding="utf-8-sig")
        except Exception as exc:  # explicit traceable failure, not a silent replacement model
            (AUDIT_DIR / "shap_status.txt").write_text(f"SHAP failed for frozen model: {exc}\n", encoding="utf-8")
    else:
        (AUDIT_DIR / "shap_status.txt").write_text(
            "SHAP not computed because the frozen selected model is not a supported tree estimator.\n",
            encoding="utf-8",
        )
    return importance


def correlation_pvalues(data: pd.DataFrame, method: str) -> pd.DataFrame:
    pvalues = pd.DataFrame(np.nan, index=data.columns, columns=data.columns, dtype=float)
    for i, first in enumerate(data.columns):
        for j, second in enumerate(data.columns):
            pair = data[[first, second]].dropna()
            if i == j:
                pvalues.loc[first, second] = 0.0
            elif len(pair) >= 4 and pair[first].nunique() > 1 and pair[second].nunique() > 1:
                if method == "spearman":
                    _, pvalue = stats.spearmanr(pair[first], pair[second])
                else:
                    _, pvalue = stats.pearsonr(pair[first], pair[second])
                pvalues.loc[first, second] = pvalue
    return pvalues


def star_label(pvalue: float) -> str:
    if not np.isfinite(pvalue):
        return ""
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return ""


def draw_diagonal_group_bracket(ax, start: int, end: int, label: str, offset: float, fontsize: float = 11.8) -> None:
    x0 = start - 0.5
    x1 = end + 0.5
    y0 = start - 1.08 - offset
    y1 = end - 0.08 - offset
    ax.plot([x0, x1], [y0, y1], color=COLORS["ink"], lw=1.35, clip_on=False)
    ax.plot([x0, x0 + 0.28], [y0, y0 - 0.20], color=COLORS["ink"], lw=1.35, clip_on=False)
    ax.plot([x1 - 0.22, x1], [y1 + 0.16, y1], color=COLORS["ink"], lw=1.35, clip_on=False)
    ax.text((x0 + x1) / 2 + 0.35, (y0 + y1) / 2 - 0.18, label, rotation=-48,
            ha="center", va="center", fontsize=fontsize, fontweight="bold", fontstyle="italic",
            color=COLORS["ink"], clip_on=False)


def draw_lower_triangle_corr_heatmap(
    ax,
    correlation: pd.DataFrame,
    pvalues: pd.DataFrame,
    group_labels: list[str],
    cmap,
    colorbar_label: str,
) -> None:
    labels = list(correlation.columns)
    count = len(labels)
    norm = plt.Normalize(-1, 1)
    for i, row_label in enumerate(labels):
        for j, column_label in enumerate(labels):
            if j > i:
                continue
            value = correlation.loc[row_label, column_label]
            if not np.isfinite(value):
                facecolor = "white"
                value_text = "NA"
                text_color = COLORS["ink"]
            else:
                facecolor = cmap(norm(value))
                value_text = f"{value:.2f}"
                text_color = "white" if abs(value) > 0.68 else COLORS["ink"]
            ax.add_patch(patches.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=facecolor, edgecolor="white", linewidth=0.95))
            ax.text(j, i - 0.07, value_text, ha="center", va="center", fontsize=10.0, fontweight="bold", color=text_color)
            significance = star_label(pvalues.loc[row_label, column_label])
            if significance and i != j:
                ax.text(j, i + 0.25, significance, ha="center", va="center", fontsize=8.8, fontweight="bold", color=text_color)
    ax.set_xlim(-0.5, count + 2.55)
    ax.set_ylim(count - 0.5, -1.05)
    ax.set_aspect("equal")
    ax.set_xticks(range(count)); ax.set_yticks(range(count))
    ax.set_xticklabels(labels, rotation=90, ha="center", va="top", fontsize=11.4, fontweight="bold")
    ax.set_yticklabels(labels, fontsize=11.4, fontweight="bold")
    ax.tick_params(axis="both", which="major", length=7.0, width=1.55, colors=COLORS["ink"], direction="out")
    for spine in ax.spines.values():
        spine.set_visible(False)
    names = pd.Series(group_labels, index=labels)
    bracket_labels = {"Target": "Output", "History": "Historical biogas", "Operation": "Operational conditions", "Weather": "Weather context"}
    start = 0; bracket_index = 0
    while start < count:
        group_name = names.iloc[start]
        end = start
        while end + 1 < count and names.iloc[end + 1] == group_name:
            end += 1
        if group_name != "Target":
            draw_diagonal_group_bracket(ax, start, end, bracket_labels[group_name], 0.18 + bracket_index * 0.10)
        start = end + 1; bracket_index += 1
    scalar = plt.cm.ScalarMappable(cmap=cmap, norm=norm); scalar.set_array([])
    cax = ax.inset_axes([0.62, 0.53, 0.31, 0.04], transform=ax.transAxes)
    colorbar = plt.colorbar(scalar, cax=cax, orientation="horizontal")
    colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    colorbar.ax.tick_params(labelsize=11.2, width=1.15, length=4.5)
    for tick in colorbar.ax.get_xticklabels():
        tick.set_fontweight("bold")
    colorbar.set_label(colorbar_label, fontsize=11.5, fontweight="bold", labelpad=4)


def create_figures(
    eligible: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    reactor_metrics: pd.DataFrame,
    coverage: pd.DataFrame,
    sensitivity: pd.DataFrame,
    importance: pd.DataFrame,
    selected_model: str,
) -> None:
    configure_style()
    reactor_order = [name for name in REACTORS if name in set(eligible["reactor_id"].dropna())]
    reactor_palette = {name: REACTOR_COLORS[name] for name in reactor_order}
    split_colors = {"train": "#EDF4FC", "validation": "#F2F7EC", "calibration": "#FFF5DD", "test": "#FBEAEA"}

    # Figure 1 retains the original distribution/reactor/time/temperature logic,
    # but every panel is recalculated from the source-audited target.
    fig, axes = plt.subplots(2, 2, figsize=(14.6, 8.6))
    fig.subplots_adjust(hspace=0.42, wspace=0.24)
    ax = axes[0, 0]
    ledger = pd.read_csv(AUDIT_DIR / "eligibility_ledger.csv")
    ledger[DATE_COL] = pd.to_datetime(ledger[DATE_COL])
    source_positive = ledger.loc[pd.to_numeric(ledger[TARGET], errors="coerce") > 0, TARGET].astype(float)
    primary_positive = eligible.loc[eligible[TARGET] > 0, TARGET].astype(float)
    combined_positive = pd.concat([source_positive, primary_positive], ignore_index=True)
    log_bins = np.logspace(np.log10(combined_positive.min()), np.log10(combined_positive.max()), 33)
    ax.hist(source_positive, bins=log_bins, histtype="step", color=COLORS["coral"], lw=2.0, label="Positive source intervals")
    ax.hist(primary_positive, bins=log_bins, color=COLORS["green"], alpha=0.74, edgecolor="white", linewidth=0.8, label="Primary source-valid")
    ax.set_xscale("log")
    ax.legend(frameon=False, loc="upper left")
    style_axis(ax, r"Biogas rate at STP (mL d$^{-1}$)", "Frequency")

    ax = axes[0, 1]
    sns.violinplot(data=eligible, x="reactor_id", y=TARGET, order=reactor_order, palette=reactor_palette, hue="reactor_id", legend=False, inner=None, linewidth=0.8, cut=0, ax=ax)
    sns.boxplot(data=eligible, x="reactor_id", y=TARGET, order=reactor_order, width=0.18, showcaps=True,
                boxprops={"facecolor": "white", "edgecolor": COLORS["ink"], "linewidth": 1.2},
                medianprops={"color": COLORS["red"], "linewidth": 1.7},
                whiskerprops={"color": COLORS["ink"], "linewidth": 1.1},
                capprops={"color": COLORS["ink"], "linewidth": 1.1}, showfliers=False, ax=ax)
    ax.set_yscale("symlog", linthresh=100)
    ax.set_xticks(np.arange(len(reactor_order)))
    ax.set_xticklabels(reactor_tick_labels(reactor_order))
    style_axis(ax, "", r"Biogas rate at STP (mL d$^{-1}$)")

    ax = axes[1, 0]
    source_context = ledger.loc[pd.to_numeric(ledger[TARGET], errors="coerce") > 0].copy()
    for reactor in reactor_order:
        raw_subset = source_context.loc[source_context["reactor_id"] == reactor].sort_values(DATE_COL)
        subset = eligible.loc[eligible["reactor_id"] == reactor].sort_values(DATE_COL)
        ax.scatter(raw_subset[DATE_COL], raw_subset[TARGET], s=27, facecolors="white", edgecolors=reactor_palette[reactor], alpha=0.54, linewidths=1.15, zorder=2)
        ax.plot(subset[DATE_COL], subset[TARGET], lw=1.8, color=reactor_palette[reactor], alpha=0.92, label=reactor_label(reactor))
    ax.set_yscale("symlog", linthresh=100)
    for split, (start, end) in SPLIT_WINDOWS.items():
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=split_colors[split], alpha=0.34, zorder=-5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    style_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=COLORS["ink"], markeredgewidth=1.2, markersize=7, label="Positive source intervals"),
        Line2D([0], [0], color=COLORS["ink"], lw=2.0, label="Primary source-valid series"),
    ]
    reactor_handles = [Line2D([0], [0], color=reactor_palette[name], lw=2.0, label=reactor_label(name)) for name in reactor_order]
    ax.legend(handles=style_handles + reactor_handles, frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.18), borderaxespad=0, handlelength=1.8, columnspacing=0.95, handletextpad=0.45)
    style_axis(ax, "Date", r"Biogas rate at STP (mL d$^{-1}$)")

    ax = axes[1, 1]
    for reactor in reactor_order:
        subset = eligible.loc[eligible["reactor_id"] == reactor]
        ax.scatter(subset["gas_temperature_C"], subset[TARGET], s=48, facecolors="none", edgecolors=reactor_palette[reactor], alpha=0.75, linewidths=1.35, label=reactor_label(reactor))
    relation = eligible[["gas_temperature_C", TARGET]].dropna()
    relation = relation.loc[relation[TARGET] >= 0]
    if len(relation) > 2:
        coefficient = np.polyfit(relation["gas_temperature_C"], np.log1p(relation[TARGET]), 1)
        line_x = np.linspace(relation["gas_temperature_C"].min(), relation["gas_temperature_C"].max(), 100)
        ax.plot(line_x, np.expm1(np.polyval(coefficient, line_x)), color=COLORS["ink"], lw=2.0)
    ax.set_yscale("symlog", linthresh=100)
    style_axis(ax, "Gas temperature (°C)", r"Biogas rate at STP (mL d$^{-1}$)")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.24), handletextpad=0.6, columnspacing=1.2)
    label_panels(axes)
    save_composite_figure(fig, "Fig01_locked_split", axes)

    # Figure 2 retains the original association-analysis structure, recalculated
    # on training data only to avoid using later outcomes for feature screening.
    groups = {
        "History": ["biogas_lag1", "biogas_lag2", "biogas_roll3", "biogas_roll7", "biogas_delta1"],
        "Operation": ["manure_fed_kg", "water_kg"],
        "Weather": ["air_temp_in_situ_lag1", "daily_mean_air_temp_lag1", "daily_solar_lag1", "daily_precip_lag1", "daily_vpd_lag1", "daily_temp_range_lag1"],
    }
    groups = {group: [name for name in names if name in eligible.columns] for group, names in groups.items()}
    assoc_features = [name for group in groups.values() for name in group]
    assoc = eligible.loc[eligible["split"] == "train", [TARGET] + assoc_features].copy()
    assoc[assoc_features] = assoc[assoc_features].apply(pd.to_numeric, errors="coerce")
    assoc_features = [name for name in assoc_features if assoc[name].notna().any()]
    assoc = assoc[[TARGET] + assoc_features]
    short = {
        TARGET: "Biogas_STP", "biogas_lag1": "Biogas_lag1", "biogas_lag2": "Biogas_lag2",
        "biogas_roll3": "Biogas_roll3", "biogas_roll7": "Biogas_roll7", "biogas_delta1": "Biogas_delta1",
        "manure_fed_kg": "Manure", "water_kg": "Water", "air_temp_in_situ_lag1": "T_local",
        "daily_mean_air_temp_lag1": "T_mean", "daily_solar_lag1": "Solar", "daily_precip_lag1": "Precip",
        "daily_vpd_lag1": "VPD", "daily_temp_range_lag1": "T_range",
    }
    display = assoc.rename(columns=short)
    group_labels = ["Target"] + [group for group, names in groups.items() for name in names if name in assoc_features]
    pearson = display.corr(method="pearson")
    spearman = display.corr(method="spearman")
    pearson_p = correlation_pvalues(display, "pearson")
    spearman_p = correlation_pvalues(display, "spearman")
    fig = plt.figure(figsize=(19.2, 13.4))
    grid = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.35, 0.82], hspace=0.42, wspace=0.16)
    axes = np.array([[fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])], [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]])
    cmap = sns.color_palette("Spectral_r", as_cmap=True)
    draw_lower_triangle_corr_heatmap(axes[0, 0], pearson, pearson_p, group_labels, cmap, "Pearson correlation coefficients (r)")
    draw_lower_triangle_corr_heatmap(axes[0, 1], spearman, spearman_p, group_labels, cmap, "Spearman correlation coefficients (rho)")

    imputed = assoc[assoc_features].fillna(assoc[assoc_features].median())
    mi = mutual_info_regression(imputed, assoc[TARGET], random_state=RANDOM_STATE)
    feature_group = {name: group for group, names in groups.items() for name in names}
    spearman_target = assoc[[TARGET] + assoc_features].corr(method="spearman")[TARGET].drop(TARGET)
    mi_frame = pd.DataFrame({
        "feature": assoc_features,
        "display": [short[name] for name in assoc_features],
        "mutual_information": mi,
        "spearman": [spearman_target.get(name, np.nan) for name in assoc_features],
        "group": [feature_group[name] for name in assoc_features],
    }).sort_values("mutual_information")
    ax = axes[1, 0]
    group_colors = {"History": COLORS["blue"], "Operation": COLORS["green"], "Weather": COLORS["coral"]}
    ax.barh(mi_frame["display"], mi_frame["mutual_information"], color=[group_colors[value] for value in mi_frame["group"]], edgecolor="white", linewidth=1.0)
    for y_index, row in enumerate(mi_frame.itertuples()):
        association_size = 0.0 if pd.isna(row.spearman) else min(abs(row.spearman), 1.0)
        ax.scatter(row.mutual_information, y_index, s=95 + 520 * association_size, facecolors="white", edgecolors=COLORS["ink"], linewidths=1.2, zorder=3)
    style_axis(ax, "Mutual information with biogas", ""); numeric_ticks(ax, x=True, y=False, n=5)

    ax = axes[1, 1]
    summary = mi_frame.groupby("group").agg(mean_abs_spearman=("spearman", lambda values: np.nanmean(np.abs(values))), mean_mi=("mutual_information", "mean"), features=("feature", "count")).reindex(["History", "Operation", "Weather"]).dropna()
    xpos = np.arange(len(summary))
    edgecolors = [group_colors[name] for name in summary.index]
    ax.scatter(xpos, summary["mean_abs_spearman"], s=850 * (summary["mean_mi"] / (summary["mean_mi"].max() + 1e-12) + 0.25), facecolors="white", edgecolors=edgecolors, linewidths=3.0)
    ax.plot(xpos, summary["mean_abs_spearman"], color=COLORS["grey"], lw=1.4, alpha=0.7)
    for x_index, row in enumerate(summary.itertuples()):
        ax.text(x_index, row.mean_abs_spearman + 0.025, f"n={int(row.features)}", ha="center", fontsize=16.0, fontweight="bold", color=COLORS["ink"])
    ax.set_xticks(xpos); ax.set_xticklabels(summary.index, fontweight="bold")
    ax.set_ylim(0, min(1.0, summary["mean_abs_spearman"].max() + 0.16))
    style_axis(ax, "", "Mean |Spearman rho| with biogas"); numeric_ticks(ax, x=False, y=True, n=5)
    label_panels(axes)
    save_composite_figure(fig, "Fig02_association_analysis", axes)

    # Figure 3: validation evidence only; panel a alone defines selection.
    order = validation_metrics.sort_values("RMSE_mL")["model"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(15.4, 10.0))
    fig.subplots_adjust(hspace=0.38, wspace=0.28)
    metrics = [
        ("RMSE_mL", r"Validation RMSE (mL d$^{-1}$)"),
        ("MAE_mL", r"Validation MAE (mL d$^{-1}$)"),
        ("R2", r"Validation $R^2$"),
    ]
    for ax, (column, xlabel) in zip(axes.reshape(-1)[:3], metrics):
        values = validation_metrics.set_index("model").loc[order, column]
        bars = ax.barh(order, values, color=[MODEL_COLORS.get(name, COLORS["grey"]) for name in order], alpha=0.92, edgecolor="white", linewidth=0.8)
        for name, bar in zip(order, bars):
            if name == selected_model:
                bar.set_edgecolor(COLORS["ink"]); bar.set_linewidth(2.0)
        if column == "R2":
            ax.axvline(0, color=COLORS["ink"], lw=1.2, ls="--")
        ax.invert_yaxis(); style_axis(ax, xlabel, ""); numeric_ticks(ax, x=True, y=False, n=5)
    holm = pd.read_csv(TABLE_DIR / "Table_S4_Holm_pairwise_validation.csv")
    holm = holm.sort_values("p_holm", ascending=True)
    ax = axes[1, 1]
    ypos = np.arange(len(holm))
    ax.hlines(ypos, 0, holm["p_holm"], color="#D6DDE2", lw=4)
    ax.scatter(holm["p_holm"], ypos, s=180, c=[MODEL_COLORS.get(name, COLORS["grey"]) for name in holm["comparison_model"]], edgecolors="white", linewidths=1.4, zorder=3)
    ax.set_yticks(ypos); ax.set_yticklabels(holm["comparison_model"]); ax.invert_yaxis()
    ax.axvline(0.05, color=COLORS["red"], lw=1.2, ls="--")
    ax.set_xlim(0, 1.03)
    style_axis(ax, "Holm-adjusted validation p value", ""); numeric_ticks(ax, x=True, y=False, n=5)
    label_panels(axes)
    save_composite_figure(fig, "Fig03_validation_selection", axes)

    # Figure 4: final test predictions, temporal residuals, errors and baselines.
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.4))
    fig.subplots_adjust(hspace=0.32, wspace=0.28)
    ax = axes[0, 0]
    for reactor in reactor_order:
        subset = predictions.loc[predictions["reactor_id"] == reactor]
        ax.scatter(
            subset["observed_mL"],
            subset["predicted_mL"],
            s=48,
            facecolors="none",
            edgecolors=[reactor_palette[reactor]],
            linewidths=1.35,
            alpha=0.80,
            label=reactor_label(reactor),
        )
    low = min(predictions["observed_mL"].min(), predictions["predicted_mL"].min())
    high = max(predictions["observed_mL"].max(), predictions["predicted_mL"].max())
    ax.fill_between([low, high], [low * 0.9, high * 0.9], [low * 1.1, high * 1.1], color=COLORS["blue"], alpha=0.08, zorder=0)
    ax.plot([low, high], [low, high], color=COLORS["grey"], lw=1.5, ls="--")
    ax.set_xscale("symlog", linthresh=100)
    ax.set_yscale("symlog", linthresh=100)
    style_axis(ax, r"Observed biogas rate at STP (mL d$^{-1}$)", r"Predicted biogas rate at STP (mL d$^{-1}$)")

    ax = axes[0, 1]
    for reactor in reactor_order:
        subset = predictions.loc[predictions["reactor_id"] == reactor]
        residual = subset["observed_mL"] - subset["predicted_mL"]
        ax.plot(subset[DATE_COL], residual, marker="o", ms=5.0, mfc="white", mew=1.2, lw=1.7, color=reactor_palette[reactor], label=reactor_label(reactor))
    ax.axhline(0, color=COLORS["ink"], lw=1.2, ls="--")
    ax.set_yscale("symlog", linthresh=100)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    style_axis(ax, "Test date", r"Residual (mL d$^{-1}$)")

    ax = axes[1, 0]
    error_frame = predictions.assign(absolute_error=np.abs(predictions["observed_mL"] - predictions["predicted_mL"]))
    sns.violinplot(data=error_frame, x="reactor_id", y="absolute_error", order=reactor_order, hue="reactor_id", palette=reactor_palette, legend=False, inner=None, linewidth=0.8, cut=0, ax=ax)
    sns.boxplot(data=error_frame, x="reactor_id", y="absolute_error", order=reactor_order, width=0.18, showfliers=True,
                boxprops={"facecolor": "white", "edgecolor": COLORS["ink"], "linewidth": 1.2},
                medianprops={"color": COLORS["red"], "linewidth": 1.7}, whiskerprops={"color": COLORS["ink"], "linewidth": 1.1},
                capprops={"color": COLORS["ink"], "linewidth": 1.1}, ax=ax)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(reactor_order)))
    ax.set_xticklabels(reactor_tick_labels(reactor_order))
    style_axis(ax, "", r"Absolute error (mL d$^{-1}$)")

    ax = axes[1, 1]
    baselines = pd.read_csv(TABLE_DIR / "Table_S8_baselines.csv").sort_values("RMSE_mL_d")
    baseline_labels = baselines["model"].str.replace("Selected ML model", selected_model)
    ybase = np.arange(len(baselines))
    ax.hlines(ybase, 0, baselines["RMSE_mL_d"], color="#D6DDE2", lw=4)
    ax.scatter(baselines["RMSE_mL_d"], ybase, s=190, c=[COLORS["lavender"] if name == "Selected ML model" else COLORS["grey"] for name in baselines["model"]], edgecolors="white", linewidths=1.4, zorder=3)
    ax.set_yticks(ybase); ax.set_yticklabels(baseline_labels); ax.invert_yaxis()
    style_axis(ax, r"Final-test RMSE (mL d$^{-1}$)", ""); numeric_ticks(ax, x=True, y=False, n=5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.015), handletextpad=0.6, columnspacing=1.3)
    label_panels(axes)
    save_composite_figure(fig, "Fig04_test_prediction", axes)

    # Figure 5: empirical split-calibration bands shown separately by reactor.
    fig, axes = plt.subplots(4, 1, figsize=(14.2, 10.6), sharex=False)
    fig.subplots_adjust(hspace=0.38)
    for ax, reactor in zip(np.asarray(axes).reshape(-1), reactor_order):
        subset = predictions.loc[predictions["reactor_id"] == reactor].sort_values(DATE_COL)
        dates = pd.to_datetime(subset[DATE_COL]).to_numpy()
        ax.fill_between(dates, subset["lower95_mL"].to_numpy(), subset["upper95_mL"].to_numpy(), color=COLORS["lavender"], alpha=0.16, linewidth=0, label="95% band")
        ax.fill_between(dates, subset["lower90_mL"].to_numpy(), subset["upper90_mL"].to_numpy(), color=COLORS["blue"], alpha=0.18, linewidth=0, label="90% band")
        ax.plot(dates, subset["observed_mL"], color=reactor_palette[reactor], marker="o", ms=5.0, mfc="white", mew=1.2, lw=2.0, label="Observed")
        ax.plot(dates, subset["predicted_mL"], color=COLORS["ink"], marker="s", ms=4.0, mfc="white", mew=1.0, lw=1.7, label="Predicted")
        ax.set_yscale("symlog", linthresh=100)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.text(0.055, 1.06, reactor_label(reactor), transform=ax.transAxes, fontsize=14, fontweight="bold", color=reactor_palette[reactor], va="bottom")
        # The same legend on each panel also makes exported panels self-contained.
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, ncol=4, frameon=False, loc="lower right",
                  bbox_to_anchor=(1.0, 1.025), borderaxespad=0, columnspacing=1.1)
        style_axis(ax, "", r"Biogas rate (mL d$^{-1}$)")
        ax.yaxis.label.set_size(15.0)
        ax.yaxis.labelpad = 3
    axes[-1].set_xlabel("Final-test date", fontweight="bold")
    label_panels(axes, x=0.005, y=1.24)
    save_composite_figure(fig, "Fig05_calibration_coverage", axes)

    # Figure 6 retains the original residual/reliability diagnostic role, with
    # unsupported Y-randomisation/domain claims replaced by prespecified peak
    # and gap diagnostics from the frozen final predictions.
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.4))
    fig.subplots_adjust(hspace=0.32, wspace=0.28, bottom=0.16)
    diag = predictions.copy()
    diag["residual_mL"] = diag["observed_mL"] - diag["predicted_mL"]
    diag["mean_observed_predicted_mL"] = (diag["observed_mL"] + diag["predicted_mL"]) / 2
    ax = axes[0, 0]
    sns.violinplot(data=diag, x="reactor_id", y="residual_mL", order=reactor_order, hue="reactor_id", palette=reactor_palette, legend=False, inner=None, linewidth=0.8, cut=0, ax=ax)
    sns.stripplot(data=diag, x="reactor_id", y="residual_mL", order=reactor_order, color=COLORS["ink"], size=3.5, alpha=0.35, ax=ax)
    ax.axhline(0, color=COLORS["ink"], lw=1.4, ls="--")
    ax.set_yscale("symlog", linthresh=100)
    ax.set_xticks(np.arange(len(reactor_order)))
    ax.set_xticklabels(reactor_tick_labels(reactor_order))
    style_axis(ax, "", r"Residual (mL d$^{-1}$)")

    ax = axes[0, 1]
    for reactor in reactor_order:
        subset = diag.loc[diag["reactor_id"] == reactor]
        ax.scatter(subset["mean_observed_predicted_mL"], subset["residual_mL"], s=44, facecolors="none", edgecolors=reactor_palette[reactor], linewidths=1.3, alpha=0.78, label=reactor_label(reactor))
    bias = diag["residual_mL"].mean(); limits = 1.96 * diag["residual_mL"].std(ddof=1)
    ax.axhline(bias, color=COLORS["ink"], lw=1.1)
    ax.axhline(bias + limits, color=COLORS["red"], lw=1.2, ls="--")
    ax.axhline(bias - limits, color=COLORS["red"], lw=1.2, ls="--")
    ax.set_xscale("symlog", linthresh=100); ax.set_yscale("symlog", linthresh=100)
    ax.margins(x=0.07)
    style_axis(ax, r"Mean observed and predicted rate (mL d$^{-1}$)", r"Observed − predicted (mL d$^{-1}$)")

    peak = pd.read_csv(TABLE_DIR / "Table_S9_peak_diagnostics.csv")
    peak_plot = peak.assign(abs_bias_mL_d=peak["mean_bias_mL_d"].abs()).melt(id_vars="subset", value_vars=["MAE_mL_d", "abs_bias_mL_d"], var_name="metric", value_name="value")
    peak_plot["metric"] = peak_plot["metric"].replace({"MAE_mL_d": "MAE", "abs_bias_mL_d": "Absolute bias"})
    ax = axes[1, 0]
    sns.barplot(data=peak_plot, x="subset", y="value", hue="metric", palette=[COLORS["navy"], COLORS["coral"]], ax=ax)
    ax.set_yscale("log"); ax.legend(frameon=False, title=None, loc="upper left")
    style_axis(ax, "", r"Error magnitude (mL d$^{-1}$)")

    gaps = pd.read_csv(TABLE_DIR / "Table_S10_gap_diagnostics.csv")
    gap_plot = gaps.melt(id_vars="subset", value_vars=["MAE_mL_d", "RMSE_mL_d"], var_name="metric", value_name="value")
    gap_plot["metric"] = gap_plot["metric"].replace({"MAE_mL_d": "MAE", "RMSE_mL_d": "RMSE"})
    gap_plot["subset"] = gap_plot["subset"].replace({"consecutive observations": "consecutive"})
    ax = axes[1, 1]
    sns.barplot(data=gap_plot, x="subset", y="value", hue="metric", palette=[COLORS["green"], COLORS["lavender"]], ax=ax)
    ax.set_yscale("log"); ax.tick_params(axis="x", rotation=12)
    ax.legend(frameon=False, title=None, loc="upper right")
    style_axis(ax, "", r"Error magnitude (mL d$^{-1}$)")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.015), handletextpad=0.6, columnspacing=1.4)
    label_panels(axes)
    save_composite_figure(fig, "Fig06_residual_reliability", axes)

    # Figure 8 retains the original four-panel sensitivity role, but reports
    # only the prespecified locked-pipeline scenarios supported by the revision.
    fig, axes = plt.subplots(2, 2, figsize=(15.8, 11.4))
    fig.subplots_adjust(hspace=0.54, wspace=0.30)
    sensitivity_panels = [
        ("RMSE_mL", r"Test RMSE (mL d$^{-1}$)", True),
        ("MAE_mL", r"Test MAE (mL d$^{-1}$)", True),
        ("R2", r"Test $R^2$", False),
        ("MASE", "Test MASE", True),
    ]
    scenario_order = sensitivity["scenario"].tolist()
    scenario_color_map = {"Primary": COLORS["navy"], "No weather": COLORS["coral"], "Parsimonious history + operation": COLORS["green"], "Consecutive only": COLORS["blue"], "Exclude R4": COLORS["lavender"], "3xIQR sensitivity": COLORS["red"]}
    scenario_colors = [scenario_color_map.get(value, COLORS["grey"]) for value in scenario_order]
    for ax, (column, xlabel, log_scale) in zip(axes.reshape(-1), sensitivity_panels):
        values = sensitivity.set_index("scenario").loc[scenario_order, column].to_numpy()
        ypos = np.arange(len(scenario_order))
        starts = np.full(len(values), max(np.min(values[values > 0]) * 0.72, 1e-6)) if log_scale else np.zeros(len(values))
        ax.hlines(ypos, starts, values, color="#D6DDE2", lw=4)
        ax.scatter(values, ypos, s=190, facecolors="white", edgecolors=scenario_colors, linewidths=2.2, zorder=3)
        ax.set_yticks(ypos); ax.set_yticklabels(scenario_order); ax.invert_yaxis()
        if log_scale:
            ax.set_xscale("log")
        else:
            ax.axvline(0, color=COLORS["ink"], lw=1.2, ls="--")
        style_axis(ax, xlabel, ""); numeric_ticks(ax, x=not log_scale, y=False, n=5)
    label_panels(axes)
    save_composite_figure(fig, "Fig08_sensitivity", axes)

    # Figure 7 restores the original one-row interpretation layout while using
    # only the actual frozen final pipeline.
    fig, axes = plt.subplots(1, 3, figsize=(18.2, 7.0))
    fig.subplots_adjust(wspace=0.56, bottom=0.17)
    top = importance.head(10).sort_values("permutation_importance_RMSE_increase_mL")
    ax = axes[0]
    ax.barh(top["feature"], top["permutation_importance_RMSE_increase_mL"], xerr=top["importance_sd_mL"], color=COLORS["navy"], alpha=0.88)
    ax.axvline(0, color=COLORS["ink"], lw=0.9, ls="--")
    style_axis(ax, r"Permutation RMSE increase (mL d$^{-1}$)", "")

    shap_frame = pd.read_csv(TABLE_DIR / "Table_S13_frozen_model_SHAP.csv")
    shap_frame["feature"] = shap_frame["transformed_feature"].str.replace(r"^(num__|cat__)", "", regex=True)
    top_shap = shap_frame.nlargest(10, "mean_abs_SHAP_mL_d").sort_values("mean_abs_SHAP_mL_d")
    ax = axes[1]
    ax.barh(top_shap["feature"], top_shap["mean_abs_SHAP_mL_d"], color=COLORS["green"], alpha=0.88)
    style_axis(ax, r"Mean absolute SHAP value (mL d$^{-1}$)", "")

    def group_feature(value: str) -> str:
        plain = re.sub(r"^(num__|cat__)", "", value)
        if plain.startswith("biogas_"):
            return "History"
        if plain in {"manure_fed_kg", "water_kg"}:
            return "Operation"
        if plain == "reactor_id" or plain.startswith("reactor_id_"):
            return "Reactor"
        if plain in {"days_since_prev"}:
            return "Timing"
        return "Weather"

    perm_group = importance.assign(group=importance["feature"].map(group_feature)).groupby("group")["permutation_importance_RMSE_increase_mL"].apply(lambda s: s.clip(lower=0).sum())
    shap_group = shap_frame.assign(group=shap_frame["feature"].map(group_feature)).groupby("group")["mean_abs_SHAP_mL_d"].sum()
    groups = sorted(set(perm_group.index).union(shap_group.index))
    grouped = pd.DataFrame({"Permutation": perm_group.reindex(groups, fill_value=0), "SHAP": shap_group.reindex(groups, fill_value=0)})
    grouped = grouped.div(grouped.sum(axis=0), axis=1).mul(100)
    ax = axes[2]
    grouped.plot(kind="bar", color=[COLORS["navy"], COLORS["green"]], ax=ax)
    ax.legend(frameon=False, loc="upper right", ncol=1)
    style_axis(ax, "Predictor group", "Within-method contribution (%)")
    label_panels(axes)
    save_composite_figure(fig, "Fig07_frozen_model_importance", axes)








