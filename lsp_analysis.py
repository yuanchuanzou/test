#!/usr/bin/env python3
"""Compute Lomb-Scargle periodograms for DM and RM columns in FRB20220529_TableS5.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt


KNOWN_TIME_COLUMNS = ["MJD", "mjd", "MJD_time"]


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(f"Could not find any of columns {candidates} in CSV header: {list(df.columns)}")


def find_error_column(df: pd.DataFrame, base: str) -> str | None:
    candidates = [
        f"{base}_err",
        f"{base}_error",
        f"{base}err",
        f"{base}error",
        f"{base}_sigma",
    ]
    for name in candidates:
        if name in df.columns:
            return name
    return None


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

    df = pd.read_csv(args.csv)
    time_col = pick_column(df, KNOWN_TIME_COLUMNS)

    for required in ["DM", "RM"]:
        if required not in df.columns:
            raise ValueError(f"Missing required column '{required}' in CSV header: {list(df.columns)}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    time = df[time_col].to_numpy(dtype=float)

    for quantity in ["DM", "RM"]:
        values = df[quantity].to_numpy(dtype=float)
        error_col = find_error_column(df, quantity)
        errors = df[error_col].to_numpy(dtype=float) if error_col else None

        period_unweighted, power_unweighted = lomb_scargle_periodogram(time, values, None)
        period_weighted, power_weighted = lomb_scargle_periodogram(time, values, errors)

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
