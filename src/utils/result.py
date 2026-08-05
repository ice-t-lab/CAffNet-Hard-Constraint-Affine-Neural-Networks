"""Reusable helpers for aggregating saved result files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.base.result import combineDfs, df2latex


MetricRename = dict[str, tuple[str, str]]
ColumnRename = dict[str, str]


def discover_seeds(run_dir: str | Path) -> list[int]:
    """Return sorted integer seeds from ``seed_*`` folders."""
    run_dir = Path(run_dir)
    seeds = []
    for path in run_dir.glob("seed_*"):
        if path.is_dir():
            try:
                seeds.append(int(path.name.removeprefix("seed_")))
            except ValueError:
                pass
    if not seeds:
        raise FileNotFoundError(f"No seed folders found in {run_dir}")
    return sorted(seeds)


def build_average_table(
    run_dir: str | Path,
    seeds: list[int],
    methods: list[str],
    output_dir: str | Path | None = None,
    metric_rename: MetricRename | None = None,
    column_rename: ColumnRename | None = None,
    scientific_columns: list[str] | None = None,
    bold_best: bool = False,
    csv_filename: str = "evaluation_all_seeds.csv",
    tex_filename: str = "evaluation_all_seeds.tex",
) -> pd.DataFrame:
    """Build mean/std table across seed metric CSVs."""
    run_dir = Path(run_dir)
    output_dir = run_dir if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not run_dir.is_dir():
        available = sorted(path.name for path in run_dir.parent.glob("*") if path.is_dir())
        raise FileNotFoundError(
            f"Missing run directory: {run_dir}. Available runs: {available}"
        )

    seed_tables = []
    for seed in seeds:
        seed_dir = run_dir / f"seed_{seed}"
        if not seed_dir.is_dir():
            available = sorted(path.name for path in run_dir.glob("seed_*") if path.is_dir())
            raise FileNotFoundError(
                f"Missing seed directory: {seed_dir}. Available seeds: {available}"
            )
        print(f"[INFO] Processing {seed_dir}")
        table = load_seed_metrics(seed_dir, methods)
        table.columns = pd.MultiIndex.from_tuples(
            [metric_header(column, metric_rename) for column in table.columns]
        )
        seed_tables.append(table)

    table = combineDfs(seed_tables, methods)
    if column_rename is not None:
        table = table.rename(columns=column_rename, level=0)
    if bold_best:
        table = bold_best_mean_values(table)
    if scientific_columns is not None:
        table = format_scientific_columns(table, scientific_columns)

    table.to_csv(output_dir / csv_filename, index=False)
    latex = df2latex(table)
    (output_dir / tex_filename).write_text(latex, encoding="utf-8")
    return table


def load_seed_metrics(seed_dir: str | Path, methods: list[str]) -> pd.DataFrame:
    """Load one seed's per-method evaluation files."""
    seed_dir = Path(seed_dir)
    tables = []
    for method in methods:
        candidates = metric_paths(seed_dir, method)
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(
                "Missing metric file for "
                f"{method} in {seed_dir}. Checked: {candidates}"
            )
        table = pd.read_csv(path)
        table["Method"] = method
        tables.append(table)
    table = pd.concat(tables, ignore_index=True)
    time_columns = ["Train Time (s)", "Test Time (s)"]
    columns = [
        column
        for column in table.columns
        if column != "Method" and column not in time_columns
    ]
    columns = ["Method", *columns]
    columns.extend(column for column in time_columns if column in table.columns)
    return table[columns]


def metric_paths(seed_dir: str | Path, method: str) -> list[Path]:
    """Return supported metric CSV paths for a method."""
    seed_dir = Path(seed_dir)
    return [
        seed_dir / method / "evaluation.csv",
        seed_dir / f"{method}_evaluation.csv",
    ]


def metric_header(
    column: str,
    metric_rename: MetricRename | None = None,
) -> tuple[str, str]:
    """Return a possibly grouped table header for one metric column."""
    if metric_rename is not None and column in metric_rename:
        return metric_rename[column]
    return (column, "")


def bold_best_mean_values(table: pd.DataFrame) -> pd.DataFrame:
    """Bold the minimum mean value in each numeric metric column."""
    out = table.copy()
    method_col = ("Method", "")
    metric_cols = [column for column in out.columns if column != method_col]

    for column in metric_cols:
        values = out[column].astype(str).str.strip()
        mean_mask = ~values.str.startswith("(")
        numeric = pd.to_numeric(
            values.loc[mean_mask].str.removesuffix("%"),
            errors="coerce",
        )
        best = numeric.min()
        if pd.isna(best):
            continue

        def fmt(value: object) -> object:
            text = str(value).strip()
            if text.startswith("("):
                return value
            is_percentage = text.endswith("%")
            try:
                numeric_value = float(text.removesuffix("%"))
            except ValueError:
                return value
            formatted = (
                f"{numeric_value:.2f}%"
                if is_percentage
                else f"{numeric_value:.4f}"
            )
            if np.isclose(numeric_value, best):
                return rf"\textbf{{{formatted}}}"
            return formatted

        out[column] = out[column].map(fmt)
    return out


def format_scientific_columns(
    table: pd.DataFrame,
    column_names: list[str],
) -> pd.DataFrame:
    """Format selected mean/std columns using rebuttal-style scientific notation."""
    out = table.copy()
    selected = {
        column
        for column in out.columns
        if column[0] in column_names
    }

    def fmt(value: object) -> object:
        text = str(value).strip()
        bold = text.startswith(r"\textbf{") and text.endswith("}")
        if bold:
            text = text[len(r"\textbf{") : -1]
        parenthesized = text.startswith("(") and text.endswith(")")
        numeric_text = text[1:-1].strip() if parenthesized else text
        try:
            numeric_value = float(numeric_text)
        except ValueError:
            return value

        if numeric_value == 0:
            formatted = "0"
        else:
            mantissa, exponent = f"{numeric_value:.4e}".split("e")
            formatted = rf"{float(mantissa):.4f}\times 10^{{{int(exponent)}}}"
        if bold:
            formatted = rf"\mathbf{{{formatted}}}"
        formatted = rf"$ {formatted} $"
        return f"({formatted})" if parenthesized else formatted

    for column in selected:
        out[column] = out[column].map(fmt)
    return out
