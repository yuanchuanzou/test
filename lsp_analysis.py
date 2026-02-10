#!/usr/bin/env python3
"""Compute Lomb-Scargle periodograms for DM and RM columns in FRB20220529_TableS5.csv."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt


KNOWN_TIME_COLUMNS = ["MJD", "mjd", "MJD_time"]
KNOWN_DM_COLUMNS = [
    "DM_stru (pc cm^-3)",
    "DM_det (pc cm^-3)",
    "DM_stru",
    "DM_det",
    "DM",
]
KNOWN_RM_COLUMNS = [
    "RM (rad m^-2)",
    "RM",
]

VALUE_ERROR_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*(?:±\s*([+-]?\d+(?:\.\d+)?))?\s*$")


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(f"Could not find any of columns {candidates} in CSV header: {list(df.columns)}")


def parse_value_error(series: pd.Series) -> tuple[np.ndarray, np.ndarray | None]:
    values: list[float] = []
    errors: list[float] = []
    has_error = False
    for item in series.astype(str):
        match = VALUE_ERROR_RE.match(item.replace("\u00b1", "±"))
        if not match:
            values.append(np.nan)
            errors.append(np.nan)
            continue
        value = float(match.group(1))
        error = match.group(2)
        values.append(value)
        if error is None:
            errors.append(np.nan)
        else:
            error_value = float(error)
            if error_value == 0:
                errors.append(np.nan)
            else:
                errors.append(error_value)
                has_error = True
    error_array = np.array(errors, dtype=float)
    return np.array(values, dtype=float), error_array if has_error else None


def read_burst_table(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8").splitlines()
    header_index = None
    header_line = None
    for idx, line in enumerate(lines):
        if line.startswith("#") and "," in line:
            header_index = idx
            header_line = line.lstrip("#").strip()
    if header_index is None or header_line is None:
        raise ValueError("Could not locate header line in CSV file.")
    columns = [col.strip() for col in header_line.split(",")]
    return pd.read_csv(path, skiprows=header_index + 1, names=columns)


def lomb_scargle_periodogram(time: np.ndarray, values: np.ndarray, errors: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    lomb = LombScargle(time, values, dy=errors)
    frequency, power = lomb.autopower()
    period = 1.0 / frequency
    return period, power


def plot_periodogram(ax: plt.Axes, period: np.ndarray, power: np.ndarray, label: str) -> None:
    ax.plot(period, power, label=label)
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("Lomb-Scargle Power")
    ax.legend()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Lomb-Scargle periodograms for DM/RM with and without errors.")
    parser.add_argument("csv", type=Path, help="Path to FRB20220529_TableS5.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for output plots")
    args = parser.parse_args()

    df = read_burst_table(args.csv)
    time_col = pick_column(df, KNOWN_TIME_COLUMNS)

    dm_col = pick_column(df, KNOWN_DM_COLUMNS)
    rm_col = pick_column(df, KNOWN_RM_COLUMNS)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    time = df[time_col].to_numpy(dtype=float)

    for quantity, column in [("DM", dm_col), ("RM", rm_col)]:
        values, errors = parse_value_error(df[column])
        mask = ~np.isnan(time) & ~np.isnan(values)
        if errors is not None:
            mask = mask & ~np.isnan(errors)
            errors = errors[mask]
        values = values[mask]
        time_masked = time[mask]

        period_unweighted, power_unweighted = lomb_scargle_periodogram(time_masked, values, None)
        period_weighted, power_weighted = lomb_scargle_periodogram(time_masked, values, errors)

        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        plot_periodogram(axes[0], period_unweighted, power_unweighted, label=f"{quantity} (unweighted)")
        plot_periodogram(axes[1], period_weighted, power_weighted, label=f"{quantity} (weighted)")
        axes[0].set_title(f"{quantity} Lomb-Scargle Periodogram")
        fig.tight_layout()

        output_path = output_dir / f"{quantity.lower()}_lsp.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

        print(f"Saved {quantity} periodogram to {output_path}")


if __name__ == "__main__":
    main()
