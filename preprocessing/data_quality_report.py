"""Data quality report: detect missing values across all preprocessed CSVs.

Scans DWD weather, Zenodo weather, and price CSVs for NaN and sentinel
values (-999). Produces a summary CSV and a combined chart (PNG) showing
which days have missing data per dataset and year.

Usage (standalone):
    python preprocessing/data_quality_report.py
    python preprocessing/data_quality_report.py --output-dir reports/quality
    python preprocessing/data_quality_report.py --dwd-dir data/weather/dwd/preprocessed

Integrated into the unified pipeline as the ``data-quality-report`` step:
    python preprocessing/data_setup.py --steps data-quality-report
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Ensure project root is importable when script is run directly
_PROJECT_ROOT_STR = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_STR)

import matplotlib.pyplot as plt
import pandas as pd

from preprocessing.utils import (
    get_measurement_columns,
    is_missing,
    parse_timestamp_column,
    parse_year_from_filename,
    resolve_path,
)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

logger = logging.getLogger(__name__)


@dataclass
class DatasetReport:
    """Quality statistics for a single CSV file."""

    dataset_name: str
    source: str
    year: int
    file_path: str
    nr_all_rows: int = 0
    nr_miss_rows: int = 0
    nr_completely_missing_days: int = 0
    nr_partially_missing_days: int = 0
    pct_completely_missing_days: float = 0.0
    pct_partially_missing_days: float = 0.0
    nr_expected_days: int = 0
    date_min: str = ""
    date_max: str = ""
    measurement_columns: str = ""
    missing_day_numbers: list[int] = field(default_factory=list)
    completely_missing_day_numbers: list[int] = field(default_factory=list)


def analyse_csv(
    csv_path: Path,
    source: str,
    check_sentinel: bool = False,
) -> DatasetReport:
    """Analyse a single CSV file for missing data.

    Args:
        csv_path: Path to the CSV file.
        source: Human-readable source name (e.g. "DWD", "Zenodo", "Price").
        check_sentinel: If True, treat -999 as missing in addition to NaN.
    """
    year = parse_year_from_filename(csv_path)
    report = DatasetReport(
        dataset_name=f"{source}/{csv_path.stem}",
        source=source,
        year=year,
        file_path=str(csv_path),
    )

    df = pd.read_csv(csv_path)
    report.nr_all_rows = len(df)

    meas_cols = get_measurement_columns(df)
    report.measurement_columns = ", ".join(meas_cols)

    if not meas_cols or df.empty:
        return report

    # Build a boolean missing-mask over measurement columns
    missing_mask = pd.DataFrame(
        {col: is_missing(df[col], check_sentinel) for col in meas_cols}
    )
    row_has_missing = missing_mask.any(axis=1)
    report.nr_miss_rows = int(row_has_missing.sum())

    # Parse timestamps and extract dates
    ts = parse_timestamp_column(df)
    report.date_min = str(ts.min().date())
    report.date_max = str(ts.max().date())

    dates = ts.dt.date
    df_analysis = pd.DataFrame({
        "date": dates,
        "row_has_missing": row_has_missing,
        "row_all_missing": missing_mask.all(axis=1),
    })

    # Group by date
    day_stats = df_analysis.groupby("date").agg(
        n_rows=("date", "size"),
        n_missing=("row_has_missing", "sum"),
        n_all_missing=("row_all_missing", "sum"),
    )

    # A day is completely missing if every row that day has all measurements missing
    completely_missing = day_stats[day_stats["n_rows"] == day_stats["n_all_missing"]]
    # A day is partially missing if it has some (but not all) missing rows
    partially_missing = day_stats[
        (day_stats["n_missing"] > 0)
        & (day_stats["n_rows"] != day_stats["n_all_missing"])
    ]

    report.nr_completely_missing_days = len(completely_missing)
    report.nr_partially_missing_days = len(partially_missing)

    # Use the full calendar year as reference for past years; for the
    # current (incomplete) year cap at today so future dates are not
    # counted as "missing".
    from datetime import date as date_cls

    year_start = pd.Timestamp(f"{year}-01-01").date()
    year_end = pd.Timestamp(f"{year}-12-31").date()
    today = date_cls.today()
    if year_end > today:
        year_end = today
    all_dates = pd.date_range(year_start, year_end, freq="D").date
    report.nr_expected_days = len(all_dates)

    present_dates = set(day_stats.index)
    absent_dates = set(all_dates) - present_dates

    # Count absent days as completely missing
    report.nr_completely_missing_days += len(absent_dates)

    # Compute day-level percentages against expected days
    expected = report.nr_expected_days or 1  # avoid division by zero
    report.pct_completely_missing_days = round(
        report.nr_completely_missing_days / expected * 100, 2
    )
    report.pct_partially_missing_days = round(
        report.nr_partially_missing_days / expected * 100, 2
    )

    # Convert to day-of-year numbers for the chart
    def _day_of_year(d: object) -> int:
        return pd.Timestamp(d).day_of_year  # type: ignore[arg-type]

    report.completely_missing_day_numbers = sorted(
        [_day_of_year(d) for d in completely_missing.index]
        + [_day_of_year(d) for d in absent_dates]
    )
    report.missing_day_numbers = sorted(
        report.completely_missing_day_numbers
        + [_day_of_year(d) for d in partially_missing.index]
    )

    return report


def discover_csvs(
    dwd_dir: str | Path,
    zenodo_weather_dir: str | Path,
    price_dir: str | Path,
) -> list[tuple[Path, str, bool]]:
    """Discover CSV files to analyse.

    Returns:
        List of (path, source_name, check_sentinel) tuples.
    """
    files: list[tuple[Path, str, bool]] = []

    dwd_path = Path(dwd_dir)
    if dwd_path.is_dir():
        for csv in sorted(dwd_path.glob("*_merged_*.csv")):
            # Skip the full (non-yearly) merged file
            if csv.stem.startswith("merged_"):
                continue
            files.append((csv, "DWD", True))

    zenodo_path = Path(zenodo_weather_dir)
    if zenodo_path.is_dir():
        for csv in sorted(zenodo_path.glob("*_weather.csv")):
            files.append((csv, "Zenodo", False))

    price_path = Path(price_dir)
    if price_path.is_dir():
        for subdir in ("awattar", "e_charts"):
            sub = price_path / subdir
            if sub.is_dir():
                for csv in sorted(sub.glob("price_data_*.csv")):
                    source_label = f"Price/{subdir}"
                    files.append((csv, source_label, False))

    return files


def generate_reports(
    dwd_dir: str | Path = "data/weather/dwd/preprocessed",
    zenodo_weather_dir: str | Path = "data/weather/zenodo/csvs_weather",
    price_dir: str | Path = "data/e_price",
) -> list[DatasetReport]:
    """Analyse all discovered CSVs and return a list of reports."""
    csv_files = discover_csvs(dwd_dir, zenodo_weather_dir, price_dir)

    if not csv_files:
        logger.warning("No CSV files found to analyse.")
        return []

    reports: list[DatasetReport] = []
    for csv_path, source, check_sentinel in csv_files:
        logger.info("Analysing %s", csv_path)
        try:
            report = analyse_csv(csv_path, source, check_sentinel)
            reports.append(report)
        except Exception as exc:
            logger.warning("Failed to analyse %s: %s", csv_path, exc)

    return reports


def write_summary_csv(reports: list[DatasetReport], output_path: Path) -> None:
    """Write the summary CSV from a list of DatasetReport objects."""
    rows = []
    for r in reports:
        pct = (r.nr_miss_rows / r.nr_all_rows * 100) if r.nr_all_rows > 0 else 0.0
        rows.append({
            "dataset_name": r.dataset_name,
            "source": r.source,
            "year": r.year,
            "nr_all_rows": r.nr_all_rows,
            "nr_miss_rows": r.nr_miss_rows,
            "pct_miss_rows": round(pct, 2),
            "nr_completely_missing_days": r.nr_completely_missing_days,
            "pct_completely_missing_days": r.pct_completely_missing_days,
            "nr_partially_missing_days": r.nr_partially_missing_days,
            "pct_partially_missing_days": r.pct_partially_missing_days,
            "nr_expected_days": r.nr_expected_days,
            "date_min": r.date_min,
            "date_max": r.date_max,
            "measurement_columns": r.measurement_columns,
            "file_path": r.file_path,
        })

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Summary CSV written to %s", output_path)


MAX_ROWS_PER_CHART: int = 18


def _paginate_reports(
    reports: list[DatasetReport],
    max_rows: int = MAX_ROWS_PER_CHART,
) -> list[list[DatasetReport]]:
    """Split reports into pages of <= max_rows entries.

    Originals and their synthesised siblings (same `source`, same `year`) are
    kept together on the same page — a group is never split across pages.
    Groups that on their own exceed `max_rows` are still emitted as a single
    page (chart row count just exceeds the cap in that case).
    """
    # Group by (source, year), preserving the input order within each group.
    groups: dict[tuple[str, int], list[DatasetReport]] = {}
    for r in reports:
        groups.setdefault((r.source, r.year), []).append(r)

    pages: list[list[DatasetReport]] = []
    current: list[DatasetReport] = []
    for key in groups:
        group = groups[key]
        if current and len(current) + len(group) > max_rows:
            pages.append(current)
            current = []
        current.extend(group)
    if current:
        pages.append(current)
    return pages


def _plot_single_chart(reports_with_data: list[DatasetReport], output_path: Path) -> None:
    """Render a single missing-days chart for the given reports."""
    labels = [f"{r.dataset_name} ({r.year})" for r in reports_with_data]
    n_rows = len(labels)

    fig_height = max(3.0, 0.45 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    bar_height = 0.6

    for i, report in enumerate(reports_with_data):
        y = n_rows - 1 - i

        # Background bar: full year extent in light grey
        max_day = 366 if report.nr_expected_days > 365 else 365
        ax.barh(y, max_day, left=0.5, height=bar_height,
                color="#e8e8e8", edgecolor="none")

        # Partially missing days (orange)
        partial_only = sorted(
            set(report.missing_day_numbers) - set(report.completely_missing_day_numbers)
        )
        if partial_only:
            for d in partial_only:
                ax.barh(y, 1, left=d - 0.5, height=bar_height,
                        color="#f0a030", edgecolor="none")

        # Completely missing days (red)
        if report.completely_missing_day_numbers:
            for d in report.completely_missing_day_numbers:
                ax.barh(y, 1, left=d - 0.5, height=bar_height,
                        color="#d03030", edgecolor="none")

        # Annotate percentages to the right of the bar
        has_any = (
            report.pct_completely_missing_days > 0
            or report.pct_partially_missing_days > 0
        )
        if has_any:
            label_parts = []
            if report.pct_completely_missing_days > 0:
                label_parts.append(f"{report.pct_completely_missing_days:.1f}% compl.")
            if report.pct_partially_missing_days > 0:
                label_parts.append(f"{report.pct_partially_missing_days:.1f}% partial")
            ax.text(
                367.5, y, "  ".join(label_parts),
                va="center", ha="left", fontsize=6.5, color="#444444",
            )

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(reversed(labels), fontsize=8)
    # Extra right margin for percentage annotations
    ax.set_xlim(0.5, 420)
    ax.set_xlabel("Day of year")
    ax.set_title("Data Quality: Missing Days per Dataset")

    # Month boundaries
    # Link: https://docs.python.org/3/library/calendar.html
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for ms in month_starts[1:]:
        ax.axvline(ms - 0.5, color="#cccccc", linewidth=0.5, zorder=0)
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels, fontsize=7)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#f0a030", label="Partially missing"),
        Patch(facecolor="#d03030", label="Completely missing"),
        Patch(facecolor="#e8e8e8", label="No missing data"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Chart saved to %s", output_path)


def plot_missing_days(reports: list[DatasetReport], output_path: Path) -> None:
    """Create one or more missing-days charts.

    Pagination: each chart contains at most ``MAX_ROWS_PER_CHART`` rows.
    Synthetic and original CSVs that share the same ``(source, year)`` are
    always placed on the same chart. Output files are named
    ``<stem>_<NN>.png`` with a 2-digit zero-padded sequence index, alongside
    ``output_path``'s suffix and parent directory.
    """
    reports_with_data = [r for r in reports if r.nr_expected_days > 0]
    if not reports_with_data:
        logger.warning("No datasets with date ranges found; skipping chart.")
        return

    pages = _paginate_reports(reports_with_data)
    base = output_path.with_suffix("")
    suffix = output_path.suffix or ".png"
    width = max(2, len(str(len(pages))))
    for idx, page in enumerate(pages, start=1):
        page_path = base.with_name(f"{base.name}_{idx:0{width}d}{suffix}")
        _plot_single_chart(page, page_path)


def run_data_quality_report(
    dwd_dir: str | Path = "data/weather/dwd/preprocessed",
    zenodo_weather_dir: str | Path = "data/weather/zenodo/csvs_weather",
    price_dir: str | Path = "data/e_price",
    output_dir: str | Path = "data/quality_reports",
) -> list[DatasetReport]:
    """Run the full data quality report pipeline.

    Called standalone or as a step from data_setup.py.
    """
    reports = generate_reports(dwd_dir, zenodo_weather_dir, price_dir)
    if not reports:
        return reports

    out = Path(output_dir)
    write_summary_csv(reports, out / "data_quality_summary.csv")
    plot_missing_days(reports, out / "missing_days_chart.png")

    # Log a quick overview
    for r in reports:
        if r.nr_miss_rows > 0 or r.nr_completely_missing_days > 0:
            logger.info(
                "  %s: %d/%d rows missing (%.4f%%), %d completely + %d partially missing days",
                r.dataset_name,
                r.nr_miss_rows,
                r.nr_all_rows,
                r.nr_miss_rows / r.nr_all_rows * 100 if r.nr_all_rows else 0,
                r.nr_completely_missing_days,
                r.nr_partially_missing_days,
            )
        else:
            logger.info("  %s: no missing data", r.dataset_name)

    return reports


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate data quality report for preprocessed CSVs"
    )
    parser.add_argument(
        "--dwd-dir",
        default="data/weather/dwd/preprocessed",
        help="Directory with DWD preprocessed CSVs",
    )
    parser.add_argument(
        "--zenodo-weather-dir",
        default="data/weather/zenodo/csvs_weather",
        help="Directory with Zenodo weather CSVs",
    )
    parser.add_argument(
        "--price-dir",
        default="data/e_price",
        help="Directory with price CSVs (awattar/ and e_charts/ subdirs)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="data/quality_reports",
        help="Output directory for report CSV and chart PNG",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    run_data_quality_report(
        dwd_dir=resolve_path(args.dwd_dir, PROJECT_ROOT),
        zenodo_weather_dir=resolve_path(args.zenodo_weather_dir, PROJECT_ROOT),
        price_dir=resolve_path(args.price_dir, PROJECT_ROOT),
        output_dir=resolve_path(args.output_dir, PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
