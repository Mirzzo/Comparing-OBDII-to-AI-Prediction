from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "artifacts" / "features"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "figures"


def main() -> None:
    split_paths = {
        "train": FEATURE_DIR / "train_features.csv",
        "validation": FEATURE_DIR / "validation_features.csv",
        "test": FEATURE_DIR / "test_features.csv",
    }

    missing = [name for name, path in split_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing cached feature tables for: "
            + ", ".join(missing)
            + ". Run `python main.py prepare-features` or `python main.py run --reuse-features` first."
        )

    class_counts = {0: 0, 1: 0}
    split_rows: list[dict[str, object]] = []
    for split_name, path in split_paths.items():
        frame = pd.read_csv(path, usecols=["class_label"])
        binary_labels = (frame["class_label"].astype(int) != 0).astype(int)
        split_count_0 = int((binary_labels == 0).sum())
        split_count_1 = int((binary_labels == 1).sum())
        class_counts[0] += split_count_0
        class_counts[1] += split_count_1
        split_rows.extend(
            [
                {"split": split_name, "class_label": 0, "count": split_count_0},
                {"split": split_name, "class_label": 1, "count": split_count_1},
            ]
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_counts_csv(
        output_path=OUTPUT_DIR / "dataset_binary_class_distribution_counts.csv",
        total_counts=class_counts,
        split_rows=split_rows,
    )
    write_chart_svg(
        output_path=OUTPUT_DIR / "dataset_binary_class_distribution.svg",
        class_counts=class_counts,
    )
    write_chart_png(
        output_path=OUTPUT_DIR / "dataset_binary_class_distribution.png",
        class_counts=class_counts,
    )


def write_counts_csv(
    output_path: Path,
    total_counts: dict[int, int],
    split_rows: list[dict[str, object]],
) -> None:
    total = sum(total_counts.values())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scope", "split", "class_label", "count", "share"])
        writer.writeheader()
        for class_label, count in sorted(total_counts.items()):
            writer.writerow(
                {
                    "scope": "all_splits",
                    "split": "all",
                    "class_label": class_label,
                    "count": count,
                    "share": round(count / total, 6),
                }
            )
        for row in split_rows:
            split_total = sum(item["count"] for item in split_rows if item["split"] == row["split"])
            writer.writerow(
                {
                    "scope": "per_split",
                    "split": row["split"],
                    "class_label": row["class_label"],
                    "count": row["count"],
                    "share": round(int(row["count"]) / split_total, 6),
                }
            )


def write_chart_svg(output_path: Path, class_counts: dict[int, int]) -> None:
    width = 900
    height = 560
    left_margin = 110
    right_margin = 60
    top_margin = 80
    bottom_margin = 95
    plot_width = width - left_margin - right_margin
    plot_height = height - top_margin - bottom_margin

    total = sum(class_counts.values())
    max_count = max(class_counts.values())
    bar_width = 170
    gap = 180
    first_bar_x = left_margin + (plot_width - ((2 * bar_width) + gap)) / 2

    bars = [
        ("Class 0", class_counts[0], "#3d5a80"),
        ("Class 1", class_counts[1], "#ee6c4d"),
    ]

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="40" text-anchor="middle" font-size="26" font-family="Cambria, Georgia, serif" font-weight="700" fill="#1f2933">Binary Class Distribution</text>',
        f'<line x1="{left_margin}" y1="{top_margin + plot_height}" x2="{width - right_margin}" y2="{top_margin + plot_height}" stroke="#30343f" stroke-width="2"/>',
        f'<line x1="{left_margin}" y1="{top_margin}" x2="{left_margin}" y2="{top_margin + plot_height}" stroke="#30343f" stroke-width="2"/>',
        f'<text x="{left_margin - 64}" y="{top_margin + plot_height / 2}" text-anchor="middle" transform="rotate(-90 {left_margin - 64},{top_margin + plot_height / 2})" font-size="16" font-family="Cambria, Georgia, serif" fill="#30343f">Samples</text>',
    ]

    for tick_index in range(6):
        tick_value = round(max_count * tick_index / 5)
        y = top_margin + plot_height - (plot_height * tick_index / 5)
        svg_lines.append(
            f'<line x1="{left_margin}" y1="{y}" x2="{width - right_margin}" y2="{y}" stroke="#ececec" stroke-width="1"/>'
        )
        svg_lines.append(
            f'<text x="{left_margin - 18}" y="{y + 5}" text-anchor="end" font-size="13" font-family="Cambria, Georgia, serif" fill="#5b6573">{tick_value}</text>'
        )

    for index, (label, count, fill) in enumerate(bars):
        x = first_bar_x + index * (bar_width + gap)
        bar_height = 0 if max_count == 0 else (count / max_count) * plot_height
        y = top_margin + plot_height - bar_height
        percentage = (count / total) * 100 if total else 0.0

        svg_lines.extend(
            [
                f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" rx="6" fill="{fill}"/>',
                f'<text x="{x + bar_width / 2}" y="{y - 14}" text-anchor="middle" font-size="18" font-family="Cambria, Georgia, serif" font-weight="700" fill="#1f2933">{count:,}</text>',
                f'<text x="{x + bar_width / 2}" y="{y - 34}" text-anchor="middle" font-size="13" font-family="Cambria, Georgia, serif" fill="#66707f">{percentage:.1f}%</text>',
                f'<text x="{x + bar_width / 2}" y="{top_margin + plot_height + 34}" text-anchor="middle" font-size="16" font-family="Cambria, Georgia, serif" fill="#1f2933">{label}</text>',
            ]
        )

    svg_lines.extend(
        [
            f'<text x="{width / 2}" y="{height - 22}" text-anchor="middle" font-size="12" font-family="Cambria, Georgia, serif" fill="#7b8794">Train + validation + test combined</text>',
            "</svg>",
        ]
    )

    output_path.write_text("\n".join(svg_lines), encoding="utf-8")


def write_chart_png(output_path: Path, class_counts: dict[int, int]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = 900
    height = 560
    left_margin = 110
    right_margin = 60
    top_margin = 80
    bottom_margin = 95
    plot_width = width - left_margin - right_margin
    plot_height = height - top_margin - bottom_margin

    total = sum(class_counts.values())
    max_count = max(class_counts.values())
    tick_values = [round(max_count * tick_index / 5) for tick_index in range(6)]

    ps_script = f"""
Add-Type -AssemblyName System.Drawing
$bmp = New-Object System.Drawing.Bitmap {width}, {height}
$graphics = [System.Drawing.Graphics]::FromImage($bmp)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
$graphics.Clear([System.Drawing.Color]::White)

$fontTitle = New-Object System.Drawing.Font('Cambria', 17, [System.Drawing.FontStyle]::Bold)
$fontNote = New-Object System.Drawing.Font('Cambria', 9)
$fontAxis = New-Object System.Drawing.Font('Cambria', 11)
$fontTick = New-Object System.Drawing.Font('Cambria', 10)
$fontValue = New-Object System.Drawing.Font('Cambria', 13, [System.Drawing.FontStyle]::Bold)
$fontPercent = New-Object System.Drawing.Font('Cambria', 10)
$fontLabel = New-Object System.Drawing.Font('Cambria', 12)

$brushBlack = [System.Drawing.Brushes]::Black
$brushDark = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(31,41,51))
$brushGray = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(91,101,115))
$brushLightGray = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(123,135,148))
$penAxis = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(48,52,63), 2)
$penGrid = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(236,236,236), 1)

$leftMargin = {left_margin}
$rightMargin = {right_margin}
$topMargin = {top_margin}
$bottomMargin = {bottom_margin}
$plotHeight = {plot_height}
$plotWidth = {plot_width}
$barWidth = 170
$gap = 180
$firstBarX = $leftMargin + (($plotWidth - ((2 * $barWidth) + $gap)) / 2)
$baselineY = $topMargin + $plotHeight
$maxCount = {max_count}
$total = {total}

$centerFormat = New-Object System.Drawing.StringFormat
$centerFormat.Alignment = [System.Drawing.StringAlignment]::Center
$centerFormat.LineAlignment = [System.Drawing.StringAlignment]::Center

$graphics.DrawString('Binary Class Distribution', $fontTitle, $brushDark, {width / 2}, 36, $centerFormat)

$graphics.DrawLine($penAxis, $leftMargin, $baselineY, {width - right_margin}, $baselineY)
$graphics.DrawLine($penAxis, $leftMargin, $topMargin, $leftMargin, $baselineY)

$graphics.TranslateTransform({left_margin - 64}, {top_margin + plot_height / 2})
$graphics.RotateTransform(-90)
$graphics.DrawString('Samples', $fontAxis, $brushDark, 0, 0, $centerFormat)
$graphics.ResetTransform()

$tickValues = @({", ".join(str(value) for value in tick_values)})
for($i = 0; $i -lt $tickValues.Count; $i++) {{
    $tickValue = [double]$tickValues[$i]
    $y = $topMargin + $plotHeight - ($plotHeight * $i / 5.0)
    $graphics.DrawLine($penGrid, $leftMargin, $y, {width - right_margin}, $y)
    $graphics.DrawString([string][int][math]::Round($tickValue), $fontTick, $brushGray, $leftMargin - 20, $y + 4, (New-Object System.Drawing.StringFormat))
}}

function Draw-Bar([float]$x, [string]$label, [int]$count, [string]$hexColor) {{
    if($maxCount -eq 0) {{
        $barHeight = 0
    }} else {{
        $barHeight = ($count / [double]$maxCount) * $plotHeight
    }}
    $y = $baselineY - $barHeight
    $percentage = if($total -eq 0) {{ 0.0 }} else {{ ($count / [double]$total) * 100.0 }}

    $color = [System.Drawing.ColorTranslator]::FromHtml($hexColor)
    $brush = New-Object System.Drawing.SolidBrush($color)
    $graphics.FillRectangle($brush, $x, $y, $barWidth, $barHeight)
    $brush.Dispose()

    $graphics.DrawString(('{{0:N0}}' -f $count), $fontValue, $brushDark, $x + ($barWidth / 2), $y - 14, $centerFormat)
    $graphics.DrawString(('{{0:N1}}%' -f $percentage), $fontPercent, $brushGray, $x + ($barWidth / 2), $y - 34, $centerFormat)
    $graphics.DrawString($label, $fontLabel, $brushDark, $x + ($barWidth / 2), $baselineY + 28, $centerFormat)
}}

Draw-Bar -x $firstBarX -label 'Class 0' -count {class_counts[0]} -hexColor '#3d5a80'
Draw-Bar -x ($firstBarX + $barWidth + $gap) -label 'Class 1' -count {class_counts[1]} -hexColor '#ee6c4d'

$graphics.DrawString('Train + validation + test combined', $fontNote, $brushLightGray, {width / 2}, {height - 22}, $centerFormat)

$bmp.Save('{str(output_path)}', [System.Drawing.Imaging.ImageFormat]::Png)

$penAxis.Dispose()
$penGrid.Dispose()
$brushDark.Dispose()
$brushGray.Dispose()
$brushLightGray.Dispose()
$fontTitle.Dispose()
$fontNote.Dispose()
$fontAxis.Dispose()
$fontTick.Dispose()
$fontValue.Dispose()
$fontPercent.Dispose()
$fontLabel.Dispose()
$graphics.Dispose()
$bmp.Dispose()
"""
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_script],
        check=True,
        cwd=str(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
