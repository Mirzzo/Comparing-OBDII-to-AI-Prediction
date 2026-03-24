from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "artifacts" / "reports" / "metrics.json"
OUTPUT_DIR = PROJECT_ROOT / "obdii_comparison" / "artifacts"
CLASS_LABELS = [0, 1, 2, 3, 4]
MODEL_SERIES = [
    ("reactive_baseline", "Reactive Baseline", "#5b8bd4"),
    ("logistic_regression", "Logistic Regression", "#ed7d31"),
    ("random_forest", "Random Forest", "#70ad47"),
]


def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find metrics at {METRICS_PATH}. Run `python main.py run` first."
        )

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    series_rows = build_series_rows(metrics)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_series_csv(
        output_path=OUTPUT_DIR / "comparison_table_test_recall_line_chart_data.csv",
        rows=series_rows,
    )
    write_chart_svg(
        output_path=OUTPUT_DIR / "comparison_table_test_recall_line_chart.svg",
        rows=series_rows,
    )
    write_chart_png(
        output_path=OUTPUT_DIR / "comparison_table_test_recall_line_chart.png",
        rows=series_rows,
    )


def build_series_rows(metrics: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_key, label, color in MODEL_SERIES:
        if model_key not in metrics:
            continue
        report = metrics[model_key]["test"]["classification_report"]
        for class_label in CLASS_LABELS:
            recall = float(report[str(class_label)]["recall"])
            rows.append(
                {
                    "model_key": model_key,
                    "model_label": label,
                    "color": color,
                    "class_label": class_label,
                    "recall": recall,
                }
            )
    return rows


def write_series_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model_key", "model_label", "class_label", "recall", "color"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_chart_svg(output_path: Path, rows: list[dict[str, object]]) -> None:
    width = 920
    height = 560
    left_margin = 90
    right_margin = 45
    top_margin = 75
    bottom_margin = 115
    plot_width = width - left_margin - right_margin
    plot_height = height - top_margin - bottom_margin

    x_step = plot_width / (len(CLASS_LABELS) - 1)
    tick_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="40" text-anchor="middle" font-size="26" font-family="Cambria, Georgia, serif" font-weight="700" fill="#1f2933">Per-Class Recall Comparison</text>',
        f'<text x="{width / 2}" y="65" text-anchor="middle" font-size="12" font-family="Cambria, Georgia, serif" fill="#6b7280">Test split from the current Comparison Table metrics</text>',
        f'<line x1="{left_margin}" y1="{top_margin + plot_height}" x2="{width - right_margin}" y2="{top_margin + plot_height}" stroke="#30343f" stroke-width="2"/>',
        f'<line x1="{left_margin}" y1="{top_margin}" x2="{left_margin}" y2="{top_margin + plot_height}" stroke="#30343f" stroke-width="2"/>',
        f'<text x="{left_margin - 55}" y="{top_margin + plot_height / 2}" text-anchor="middle" transform="rotate(-90 {left_margin - 55},{top_margin + plot_height / 2})" font-size="16" font-family="Cambria, Georgia, serif" fill="#30343f">Recall</text>',
        f'<text x="{width / 2}" y="{height - 58}" text-anchor="middle" font-size="16" font-family="Cambria, Georgia, serif" fill="#30343f">Class Label</text>',
    ]

    for tick_value in tick_values:
        y = top_margin + plot_height - (tick_value * plot_height)
        svg_lines.append(
            f'<line x1="{left_margin}" y1="{y}" x2="{width - right_margin}" y2="{y}" stroke="#d9e6f5" stroke-width="1.3"/>'
        )
        svg_lines.append(
            f'<text x="{left_margin - 16}" y="{y + 5}" text-anchor="end" font-size="13" font-family="Cambria, Georgia, serif" fill="#5b6573">{tick_value:.1f}</text>'
        )

    for index, class_label in enumerate(CLASS_LABELS):
        x = left_margin + (index * x_step)
        svg_lines.append(
            f'<text x="{x}" y="{top_margin + plot_height + 30}" text-anchor="middle" font-size="14" font-family="Cambria, Georgia, serif" fill="#1f2933">Class {class_label}</text>'
        )

    for model_key, label, color in MODEL_SERIES:
        model_points = [row for row in rows if row["model_key"] == model_key]
        if not model_points:
            continue

        point_strings = []
        for row in model_points:
            x = left_margin + (int(row["class_label"]) * x_step)
            y = top_margin + plot_height - (float(row["recall"]) * plot_height)
            point_strings.append(f"{x},{y}")
        svg_lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(point_strings)}"/>'
        )
        for row in model_points:
            x = left_margin + (int(row["class_label"]) * x_step)
            y = top_margin + plot_height - (float(row["recall"]) * plot_height)
            svg_lines.append(
                f'<circle cx="{x}" cy="{y}" r="5" fill="{color}" stroke="white" stroke-width="1.5"/>'
            )

    legend_y = height - 24
    legend_items = [(label, color) for _, label, color in MODEL_SERIES if any(r["model_label"] == label for r in rows)]
    total_legend_width = len(legend_items) * 180
    legend_x = (width - total_legend_width) / 2
    for index, (label, color) in enumerate(legend_items):
        item_x = legend_x + (index * 180)
        svg_lines.extend(
            [
                f'<line x1="{item_x}" y1="{legend_y}" x2="{item_x + 28}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
                f'<circle cx="{item_x + 14}" cy="{legend_y}" r="4.5" fill="{color}" stroke="white" stroke-width="1.2"/>',
                f'<text x="{item_x + 36}" y="{legend_y + 5}" font-size="13" font-family="Cambria, Georgia, serif" fill="#1f2933">{label}</text>',
            ]
        )

    svg_lines.append("</svg>")
    output_path.write_text("\n".join(svg_lines), encoding="utf-8")


def write_chart_png(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = 920
    height = 560
    left_margin = 90
    right_margin = 45
    top_margin = 75
    bottom_margin = 115
    plot_width = width - left_margin - right_margin
    plot_height = height - top_margin - bottom_margin
    x_step = plot_width / (len(CLASS_LABELS) - 1)

    series_payload = []
    for model_key, label, color in MODEL_SERIES:
        model_points = [row for row in rows if row["model_key"] == model_key]
        if model_points:
            series_payload.append(
                {
                    "label": label,
                    "color": color,
                    "recalls": [float(row["recall"]) for row in model_points],
                }
            )

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
$fontLabel = New-Object System.Drawing.Font('Cambria', 10)
$fontLegend = New-Object System.Drawing.Font('Cambria', 10)

$brushDark = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(31,41,51))
$brushGray = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(91,101,115))
$penAxis = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(48,52,63), 2)
$penGrid = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(217,230,245), 1.3)

$leftMargin = {left_margin}
$rightMargin = {right_margin}
$topMargin = {top_margin}
$bottomMargin = {bottom_margin}
$plotWidth = {plot_width}
$plotHeight = {plot_height}
$baselineY = $topMargin + $plotHeight
$xStep = {x_step}

$centerFormat = New-Object System.Drawing.StringFormat
$centerFormat.Alignment = [System.Drawing.StringAlignment]::Center
$centerFormat.LineAlignment = [System.Drawing.StringAlignment]::Center

$graphics.DrawString('Per-Class Recall Comparison', $fontTitle, $brushDark, {width / 2}, 36, $centerFormat)
$graphics.DrawString('Test split from the current Comparison Table metrics', $fontNote, $brushGray, {width / 2}, 60, $centerFormat)

$graphics.DrawLine($penAxis, $leftMargin, $baselineY, {width - right_margin}, $baselineY)
$graphics.DrawLine($penAxis, $leftMargin, $topMargin, $leftMargin, $baselineY)

$graphics.TranslateTransform({left_margin - 55}, {top_margin + plot_height / 2})
$graphics.RotateTransform(-90)
$graphics.DrawString('Recall', $fontAxis, $brushDark, 0, 0, $centerFormat)
$graphics.ResetTransform()
$graphics.DrawString('Class Label', $fontAxis, $brushDark, {width / 2}, {height - 58}, $centerFormat)

$tickValues = @(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
for($i = 0; $i -lt $tickValues.Count; $i++) {{
    $tickValue = [double]$tickValues[$i]
    $y = $topMargin + $plotHeight - ($tickValue * $plotHeight)
    $graphics.DrawLine($penGrid, $leftMargin, $y, {width - right_margin}, $y)
    $graphics.DrawString(($tickValue.ToString('0.0')), $fontTick, $brushGray, $leftMargin - 16, $y + 4, (New-Object System.Drawing.StringFormat))
}}

for($i = 0; $i -lt {len(CLASS_LABELS)}; $i++) {{
    $x = $leftMargin + ($i * $xStep)
    $graphics.DrawString(('Class ' + $i), $fontLabel, $brushDark, $x, $baselineY + 30, $centerFormat)
}}

function Draw-Series([string]$label, [string]$hexColor, [double[]]$recalls) {{
    $color = [System.Drawing.ColorTranslator]::FromHtml($hexColor)
    $pen = New-Object System.Drawing.Pen($color, 3)
    $brush = New-Object System.Drawing.SolidBrush($color)
    $points = New-Object 'System.Drawing.PointF[]' $recalls.Length
    for($j = 0; $j -lt $recalls.Length; $j++) {{
        $x = [float]($leftMargin + ($j * $xStep))
        $y = [float]($topMargin + $plotHeight - ($recalls[$j] * $plotHeight))
        $points[$j] = New-Object System.Drawing.PointF($x, $y)
    }}
    if($points.Length -gt 1) {{
        $graphics.DrawLines($pen, $points)
    }}
    foreach($point in $points) {{
        $graphics.FillEllipse($brush, $point.X - 5, $point.Y - 5, 10, 10)
        $graphics.DrawEllipse([System.Drawing.Pens]::White, $point.X - 5, $point.Y - 5, 10, 10)
    }}
    $pen.Dispose()
    $brush.Dispose()
}}
"""

    for series in series_payload:
        recalls_str = ", ".join(f"{value:.6f}" for value in series["recalls"])
        ps_script += (
            f"\nDraw-Series -label '{series['label']}' -hexColor '{series['color']}' "
            f"-recalls @({recalls_str})\n"
        )

    legend_y = height - 24
    legend_total_width = len(series_payload) * 180
    legend_start_x = (width - legend_total_width) / 2
    ps_script += f"""
$legendY = {legend_y}
$legendStartX = {legend_start_x}
"""
    for index, series in enumerate(series_payload):
        ps_script += f"""
$legendX{index} = $legendStartX + ({index} * 180)
$legendColor{index} = [System.Drawing.ColorTranslator]::FromHtml('{series['color']}')
$legendPen{index} = New-Object System.Drawing.Pen($legendColor{index}, 3)
$legendBrush{index} = New-Object System.Drawing.SolidBrush($legendColor{index})
$graphics.DrawLine($legendPen{index}, $legendX{index}, $legendY, $legendX{index} + 28, $legendY)
$graphics.FillEllipse($legendBrush{index}, $legendX{index} + 9.5, $legendY - 4.5, 9, 9)
$graphics.DrawEllipse([System.Drawing.Pens]::White, $legendX{index} + 9.5, $legendY - 4.5, 9, 9)
$graphics.DrawString('{series['label']}', $fontLegend, $brushDark, $legendX{index} + 36, $legendY + 4, (New-Object System.Drawing.StringFormat))
$legendPen{index}.Dispose()
$legendBrush{index}.Dispose()
"""

    ps_script += f"""
$bmp.Save('{str(output_path)}', [System.Drawing.Imaging.ImageFormat]::Png)
$penAxis.Dispose()
$penGrid.Dispose()
$brushDark.Dispose()
$brushGray.Dispose()
$fontTitle.Dispose()
$fontNote.Dispose()
$fontAxis.Dispose()
$fontTick.Dispose()
$fontLabel.Dispose()
$fontLegend.Dispose()
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
