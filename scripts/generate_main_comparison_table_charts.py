from __future__ import annotations

import csv
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "obdii_comparison" / "artifacts"

TABLE_ROWS = [
    {
        "model": "Reactive OBD-II-style baseline",
        "split": "Validation",
        "accuracy": 0.015061,
        "macro_precision": 0.003012,
        "macro_recall": 0.200000,
        "macro_f1": 0.005935,
        "weighted_f1": 0.000447,
        "mean_challenge_cost": 9.822830,
    },
    {
        "model": "Reactive OBD-II-style baseline",
        "split": "Test",
        "accuracy": 0.011893,
        "macro_precision": 0.002379,
        "macro_recall": 0.200000,
        "macro_f1": 0.004701,
        "weighted_f1": 0.000280,
        "mean_challenge_cost": 9.845590,
    },
    {
        "model": "Two-stage CatBoost",
        "split": "Validation",
        "accuracy": 0.930836,
        "macro_precision": 0.211861,
        "macro_recall": 0.222414,
        "macro_f1": 0.214681,
        "weighted_f1": 0.941371,
        "mean_challenge_cost": 9.904479,
    },
    {
        "model": "Two-stage CatBoost",
        "split": "Test",
        "accuracy": 0.891378,
        "macro_precision": 0.213186,
        "macro_recall": 0.250957,
        "macro_f1": 0.215768,
        "weighted_f1": 0.919771,
        "mean_challenge_cost": 8.954410,
    },
]

SCORE_METRICS = [
    ("accuracy", "Accuracy"),
    ("macro_precision", "Macro Precision"),
    ("macro_recall", "Macro Recall"),
    ("macro_f1", "Macro F1"),
    ("weighted_f1", "Weighted F1"),
]

SCORE_SERIES = [
    ("Reactive OBD-II-style baseline", "Validation", "Reactive baseline (Validation)", "#5b8bd4", False),
    ("Reactive OBD-II-style baseline", "Test", "Reactive baseline (Test)", "#5b8bd4", True),
    ("Two-stage CatBoost", "Validation", "Two-stage CatBoost (Validation)", "#ed7d31", False),
    ("Two-stage CatBoost", "Test", "Two-stage CatBoost (Test)", "#ed7d31", True),
]

COST_SERIES = [
    ("Reactive OBD-II-style baseline", "Reactive baseline", "#5b8bd4"),
    ("Two-stage CatBoost", "Two-stage CatBoost", "#ed7d31"),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_table_csv(OUTPUT_DIR / "main_comparison_table_values.csv")
    write_score_chart_svg(OUTPUT_DIR / "main_comparison_score_line_chart.svg")
    write_score_chart_png(OUTPUT_DIR / "main_comparison_score_line_chart.png")
    write_cost_chart_svg(OUTPUT_DIR / "main_comparison_mean_cost_line_chart.svg")
    write_cost_chart_png(OUTPUT_DIR / "main_comparison_mean_cost_line_chart.png")
    write_paper_style_score_chart_svg(
        output_path=OUTPUT_DIR / "main_comparison_validation_score_line_chart_paper_style.svg",
        split="Validation",
    )
    write_paper_style_score_chart_png(
        output_path=OUTPUT_DIR / "main_comparison_validation_score_line_chart_paper_style.png",
        split="Validation",
    )
    write_paper_style_score_chart_svg(
        output_path=OUTPUT_DIR / "main_comparison_test_score_line_chart_paper_style.svg",
        split="Test",
    )
    write_paper_style_score_chart_png(
        output_path=OUTPUT_DIR / "main_comparison_test_score_line_chart_paper_style.png",
        split="Test",
    )


def write_table_csv(output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "split",
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_f1",
                "mean_challenge_cost",
            ],
        )
        writer.writeheader()
        writer.writerows(TABLE_ROWS)


def row_for(model: str, split: str) -> dict[str, object]:
    for row in TABLE_ROWS:
        if row["model"] == model and row["split"] == split:
            return row
    raise KeyError(f"Missing row for model={model}, split={split}")


def write_score_chart_svg(output_path: Path) -> None:
    width = 1520
    height = 900
    left_margin = 135
    right_margin = 85
    top_margin = 112
    bottom_margin = 210
    plot_width = width - left_margin - right_margin
    plot_height = height - top_margin - bottom_margin
    x_step = plot_width / (len(SCORE_METRICS) - 1)
    tick_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="58" text-anchor="middle" font-size="42" font-family="Cambria, Georgia, serif" font-weight="700" fill="#1f2933">Main Comparison Score Metrics</text>',
        f'<text x="{width / 2}" y="92" text-anchor="middle" font-size="20" font-family="Cambria, Georgia, serif" fill="#6b7280">Generated from the same values used in Table III.2</text>',
        f'<line x1="{left_margin}" y1="{top_margin + plot_height}" x2="{width - right_margin}" y2="{top_margin + plot_height}" stroke="#30343f" stroke-width="2"/>',
        f'<line x1="{left_margin}" y1="{top_margin}" x2="{left_margin}" y2="{top_margin + plot_height}" stroke="#30343f" stroke-width="2"/>',
        f'<text x="{left_margin - 82}" y="{top_margin + plot_height / 2}" text-anchor="middle" transform="rotate(-90 {left_margin - 82},{top_margin + plot_height / 2})" font-size="25" font-family="Cambria, Georgia, serif" fill="#30343f">Metric value</text>',
        f'<text x="{width / 2}" y="{height - 118}" text-anchor="middle" font-size="25" font-family="Cambria, Georgia, serif" fill="#30343f">Metric</text>',
    ]

    for tick_value in tick_values:
        y = top_margin + plot_height - (tick_value * plot_height)
        svg_lines.append(
            f'<line x1="{left_margin}" y1="{y}" x2="{width - right_margin}" y2="{y}" stroke="#d9e6f5" stroke-width="1.3"/>'
        )
        svg_lines.append(
            f'<text x="{left_margin - 24}" y="{y + 7}" text-anchor="end" font-size="20" font-family="Cambria, Georgia, serif" fill="#5b6573">{tick_value:.1f}</text>'
        )

    for index, (_, metric_label) in enumerate(SCORE_METRICS):
        x = left_margin + (index * x_step)
        svg_lines.append(
            f'<text x="{x}" y="{top_margin + plot_height + 50}" text-anchor="middle" font-size="21" font-family="Cambria, Georgia, serif" fill="#1f2933">{metric_label}</text>'
        )

    for model, split, label, color, dashed in SCORE_SERIES:
        row = row_for(model, split)
        point_strings: list[str] = []
        for index, (metric_key, _) in enumerate(SCORE_METRICS):
            x = left_margin + (index * x_step)
            y = top_margin + plot_height - (float(row[metric_key]) * plot_height)
            point_strings.append(f"{x},{y}")
        dash_attr = ' stroke-dasharray="8 6"' if dashed else ""
        svg_lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3"{dash_attr} points="{" ".join(point_strings)}"/>'
        )
        for index, (metric_key, _) in enumerate(SCORE_METRICS):
            x = left_margin + (index * x_step)
            y = top_margin + plot_height - (float(row[metric_key]) * plot_height)
            svg_lines.append(
                f'<circle cx="{x}" cy="{y}" r="7" fill="{color}" stroke="white" stroke-width="1.7"/>'
            )

    legend_y = height - 48
    legend_items = [(label, color, dashed) for _, _, label, color, dashed in SCORE_SERIES]
    legend_spacing = 340
    legend_x = (width - (legend_spacing * len(legend_items))) / 2
    for index, (label, color, dashed) in enumerate(legend_items):
        item_x = legend_x + (index * legend_spacing)
        dash_attr = ' stroke-dasharray="8 6"' if dashed else ""
        svg_lines.extend(
            [
                f'<line x1="{item_x}" y1="{legend_y}" x2="{item_x + 48}" y2="{legend_y}" stroke="{color}" stroke-width="4"{dash_attr}/>',
                f'<circle cx="{item_x + 24}" cy="{legend_y}" r="6.5" fill="{color}" stroke="white" stroke-width="1.3"/>',
                f'<text x="{item_x + 60}" y="{legend_y + 7}" font-size="18" font-family="Cambria, Georgia, serif" fill="#1f2933">{label}</text>',
            ]
        )

    svg_lines.append("</svg>")
    output_path.write_text("\n".join(svg_lines), encoding="utf-8")


def write_score_chart_png(output_path: Path) -> None:
    width = 1520
    height = 900
    left_margin = 135
    right_margin = 85
    top_margin = 112
    bottom_margin = 210
    plot_width = width - left_margin - right_margin
    plot_height = height - top_margin - bottom_margin
    x_step = plot_width / (len(SCORE_METRICS) - 1)

    ps_script = f"""
Add-Type -AssemblyName System.Drawing
$bmp = New-Object System.Drawing.Bitmap {width}, {height}
$graphics = [System.Drawing.Graphics]::FromImage($bmp)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
$graphics.Clear([System.Drawing.Color]::White)

$fontTitle = New-Object System.Drawing.Font('Cambria', 28, [System.Drawing.FontStyle]::Bold)
$fontNote = New-Object System.Drawing.Font('Cambria', 14)
$fontAxis = New-Object System.Drawing.Font('Cambria', 17)
$fontTick = New-Object System.Drawing.Font('Cambria', 15)
$fontLabel = New-Object System.Drawing.Font('Cambria', 15)
$fontLegend = New-Object System.Drawing.Font('Cambria', 14)

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

$graphics.DrawString('Main Comparison Score Metrics', $fontTitle, $brushDark, {width / 2}, 52, $centerFormat)
$graphics.DrawString('Generated from the same values used in Table III.2', $fontNote, $brushGray, {width / 2}, 90, $centerFormat)
$graphics.DrawLine($penAxis, $leftMargin, $baselineY, {width - right_margin}, $baselineY)
$graphics.DrawLine($penAxis, $leftMargin, $topMargin, $leftMargin, $baselineY)

$graphics.TranslateTransform({left_margin - 82}, {top_margin + plot_height / 2})
$graphics.RotateTransform(-90)
$graphics.DrawString('Metric value', $fontAxis, $brushDark, 0, 0, $centerFormat)
$graphics.ResetTransform()
$graphics.DrawString('Metric', $fontAxis, $brushDark, {width / 2}, {height - 118}, $centerFormat)

$tickValues = @(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
for($i = 0; $i -lt $tickValues.Count; $i++) {{
    $tickValue = [double]$tickValues[$i]
    $y = $topMargin + $plotHeight - ($tickValue * $plotHeight)
    $graphics.DrawLine($penGrid, $leftMargin, $y, {width - right_margin}, $y)
    $graphics.DrawString(($tickValue.ToString('0.0')), $fontTick, $brushGray, $leftMargin - 30, $y + 4, (New-Object System.Drawing.StringFormat))
}}
"""
    for index, (_, metric_label) in enumerate(SCORE_METRICS):
        x = left_margin + (index * x_step)
        ps_script += (
            f"\n$graphics.DrawString('{metric_label}', $fontLabel, $brushDark, {x}, "
            f"{top_margin + plot_height + 50}, $centerFormat)\n"
        )

    for model, split, label, color, dashed in SCORE_SERIES:
        row = row_for(model, split)
        points = []
        for metric_key, _ in SCORE_METRICS:
            value = float(row[metric_key])
            points.append(value)
        points_str = ", ".join(f"{value:.6f}" for value in points)
        dash_str = "$true" if dashed else "$false"
        ps_script += f"""
function Draw-Score-Series-{slugify(label)}([double[]]$values) {{
    $color = [System.Drawing.ColorTranslator]::FromHtml('{color}')
    $pen = New-Object System.Drawing.Pen($color, 3)
    if({dash_str}) {{ $pen.DashPattern = @(8,6) }}
    $brush = New-Object System.Drawing.SolidBrush($color)
    $points = New-Object 'System.Drawing.PointF[]' $values.Length
    for($j = 0; $j -lt $values.Length; $j++) {{
        $x = [float]($leftMargin + ($j * $xStep))
        $y = [float]($topMargin + $plotHeight - ($values[$j] * $plotHeight))
        $points[$j] = New-Object System.Drawing.PointF($x, $y)
    }}
    if($points.Length -gt 1) {{ $graphics.DrawLines($pen, $points) }}
    foreach($point in $points) {{
        $graphics.FillEllipse($brush, $point.X - 7, $point.Y - 7, 14, 14)
        $graphics.DrawEllipse([System.Drawing.Pens]::White, $point.X - 7, $point.Y - 7, 14, 14)
    }}
    $pen.Dispose()
    $brush.Dispose()
}}
Draw-Score-Series-{slugify(label)} -values @({points_str})
"""

    legend_y = height - 48
    legend_spacing = 340
    legend_start_x = (width - (legend_spacing * len(SCORE_SERIES))) / 2
    ps_script += f"\n$legendY = {legend_y}\n$legendStartX = {legend_start_x}\n"
    for index, (_, _, label, color, dashed) in enumerate(SCORE_SERIES):
        dash_lines = f"$legendPen{index}.DashPattern = @(8,6)\n" if dashed else ""
        ps_script += f"""
$legendX{index} = $legendStartX + ({index} * {legend_spacing})
$legendColor{index} = [System.Drawing.ColorTranslator]::FromHtml('{color}')
$legendPen{index} = New-Object System.Drawing.Pen($legendColor{index}, 3)
{dash_lines}$legendBrush{index} = New-Object System.Drawing.SolidBrush($legendColor{index})
$graphics.DrawLine($legendPen{index}, $legendX{index}, $legendY, $legendX{index} + 48, $legendY)
$graphics.FillEllipse($legendBrush{index}, $legendX{index} + 17, $legendY - 6.5, 13, 13)
$graphics.DrawEllipse([System.Drawing.Pens]::White, $legendX{index} + 17, $legendY - 6.5, 13, 13)
$graphics.DrawString('{label}', $fontLegend, $brushDark, $legendX{index} + 60, $legendY + 6, (New-Object System.Drawing.StringFormat))
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


def write_cost_chart_svg(output_path: Path) -> None:
    width = 1120
    height = 720
    left_margin = 125
    right_margin = 70
    top_margin = 98
    bottom_margin = 160
    plot_width = width - left_margin - right_margin
    plot_height = height - top_margin - bottom_margin
    x_positions = {
        "Validation": left_margin + (plot_width * 0.25),
        "Test": left_margin + (plot_width * 0.75),
    }
    cost_values = [float(row["mean_challenge_cost"]) for row in TABLE_ROWS]
    min_cost = min(cost_values)
    max_cost = max(cost_values)
    padded_min = min_cost - 0.15
    padded_max = max_cost + 0.15

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="54" text-anchor="middle" font-size="38" font-family="Cambria, Georgia, serif" font-weight="700" fill="#1f2933">Mean Challenge Cost</text>',
        f'<text x="{width / 2}" y="88" text-anchor="middle" font-size="20" font-family="Cambria, Georgia, serif" fill="#6b7280">Companion chart for the same main comparison table</text>',
        f'<line x1="{left_margin}" y1="{top_margin + plot_height}" x2="{width - right_margin}" y2="{top_margin + plot_height}" stroke="#30343f" stroke-width="2"/>',
        f'<line x1="{left_margin}" y1="{top_margin}" x2="{left_margin}" y2="{top_margin + plot_height}" stroke="#30343f" stroke-width="2"/>',
        f'<text x="{left_margin - 76}" y="{top_margin + plot_height / 2}" text-anchor="middle" transform="rotate(-90 {left_margin - 76},{top_margin + plot_height / 2})" font-size="24" font-family="Cambria, Georgia, serif" fill="#30343f">Mean cost</text>',
    ]

    tick_values = [round(padded_min + ((padded_max - padded_min) * step / 4), 3) for step in range(5)]
    for tick_value in tick_values:
        y = top_margin + plot_height - (((tick_value - padded_min) / (padded_max - padded_min)) * plot_height)
        svg_lines.append(
            f'<line x1="{left_margin}" y1="{y}" x2="{width - right_margin}" y2="{y}" stroke="#d9e6f5" stroke-width="1.2"/>'
        )
        svg_lines.append(
            f'<text x="{left_margin - 22}" y="{y + 7}" text-anchor="end" font-size="19" font-family="Cambria, Georgia, serif" fill="#5b6573">{tick_value:.2f}</text>'
        )

    for split, x in x_positions.items():
        svg_lines.append(
            f'<text x="{x}" y="{top_margin + plot_height + 48}" text-anchor="middle" font-size="21" font-family="Cambria, Georgia, serif" fill="#1f2933">{split}</text>'
        )

    for model, label, color in COST_SERIES:
        points: list[str] = []
        for split in ["Validation", "Test"]:
            row = row_for(model, split)
            value = float(row["mean_challenge_cost"])
            x = x_positions[split]
            y = top_margin + plot_height - (((value - padded_min) / (padded_max - padded_min)) * plot_height)
            points.append(f"{x},{y}")
        svg_lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(points)}"/>'
        )
        for split in ["Validation", "Test"]:
            row = row_for(model, split)
            value = float(row["mean_challenge_cost"])
            x = x_positions[split]
            y = top_margin + plot_height - (((value - padded_min) / (padded_max - padded_min)) * plot_height)
            svg_lines.append(
                f'<circle cx="{x}" cy="{y}" r="7" fill="{color}" stroke="white" stroke-width="1.4"/>'
            )
            svg_lines.append(
                f'<text x="{x}" y="{y - 17}" text-anchor="middle" font-size="18" font-family="Cambria, Georgia, serif" fill="#1f2933">{value:.3f}</text>'
            )

    legend_y = height - 40
    legend_x = (width - (len(COST_SERIES) * 280)) / 2
    for index, (_, label, color) in enumerate(COST_SERIES):
        item_x = legend_x + (index * 280)
        svg_lines.extend(
            [
                f'<line x1="{item_x}" y1="{legend_y}" x2="{item_x + 48}" y2="{legend_y}" stroke="{color}" stroke-width="4"/>',
                f'<circle cx="{item_x + 24}" cy="{legend_y}" r="6.5" fill="{color}" stroke="white" stroke-width="1.2"/>',
                f'<text x="{item_x + 60}" y="{legend_y + 7}" font-size="18" font-family="Cambria, Georgia, serif" fill="#1f2933">{label}</text>',
            ]
        )

    svg_lines.append("</svg>")
    output_path.write_text("\n".join(svg_lines), encoding="utf-8")


def write_cost_chart_png(output_path: Path) -> None:
    width = 1120
    height = 720
    left_margin = 125
    right_margin = 70
    top_margin = 98
    bottom_margin = 160
    plot_width = width - left_margin - right_margin
    plot_height = height - top_margin - bottom_margin
    validation_x = left_margin + (plot_width * 0.25)
    test_x = left_margin + (plot_width * 0.75)
    cost_values = [float(row["mean_challenge_cost"]) for row in TABLE_ROWS]
    min_cost = min(cost_values)
    max_cost = max(cost_values)
    padded_min = min_cost - 0.15
    padded_max = max_cost + 0.15

    ps_script = f"""
Add-Type -AssemblyName System.Drawing
$bmp = New-Object System.Drawing.Bitmap {width}, {height}
$graphics = [System.Drawing.Graphics]::FromImage($bmp)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
$graphics.Clear([System.Drawing.Color]::White)

$fontTitle = New-Object System.Drawing.Font('Cambria', 26, [System.Drawing.FontStyle]::Bold)
$fontNote = New-Object System.Drawing.Font('Cambria', 14)
$fontAxis = New-Object System.Drawing.Font('Cambria', 16)
$fontTick = New-Object System.Drawing.Font('Cambria', 14)
$fontLabel = New-Object System.Drawing.Font('Cambria', 15)
$fontValue = New-Object System.Drawing.Font('Cambria', 13)
$fontLegend = New-Object System.Drawing.Font('Cambria', 14)

$brushDark = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(31,41,51))
$brushGray = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(91,101,115))
$penAxis = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(48,52,63), 2)
$penGrid = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(217,230,245), 1.2)

$leftMargin = {left_margin}
$topMargin = {top_margin}
$plotHeight = {plot_height}
$baselineY = $topMargin + $plotHeight
$validationX = {validation_x}
$testX = {test_x}
$minCost = {padded_min}
$maxCost = {padded_max}

$centerFormat = New-Object System.Drawing.StringFormat
$centerFormat.Alignment = [System.Drawing.StringAlignment]::Center
$centerFormat.LineAlignment = [System.Drawing.StringAlignment]::Center

$graphics.DrawString('Mean Challenge Cost', $fontTitle, $brushDark, {width / 2}, 50, $centerFormat)
$graphics.DrawString('Companion chart for the same main comparison table', $fontNote, $brushGray, {width / 2}, 86, $centerFormat)
$graphics.DrawLine($penAxis, $leftMargin, $baselineY, {width - right_margin}, $baselineY)
$graphics.DrawLine($penAxis, $leftMargin, $topMargin, $leftMargin, $baselineY)

$graphics.TranslateTransform({left_margin - 76}, {top_margin + plot_height / 2})
$graphics.RotateTransform(-90)
$graphics.DrawString('Mean cost', $fontAxis, $brushDark, 0, 0, $centerFormat)
$graphics.ResetTransform()

$tickValues = @({", ".join(f"{round(padded_min + ((padded_max - padded_min) * step / 4), 3):.3f}" for step in range(5))})
for($i = 0; $i -lt $tickValues.Count; $i++) {{
    $tickValue = [double]$tickValues[$i]
    $y = $topMargin + $plotHeight - ((($tickValue - $minCost) / ($maxCost - $minCost)) * $plotHeight)
    $graphics.DrawLine($penGrid, $leftMargin, $y, {width - right_margin}, $y)
    $graphics.DrawString(($tickValue.ToString('0.00')), $fontTick, $brushGray, $leftMargin - 24, $y + 4, (New-Object System.Drawing.StringFormat))
}}

$graphics.DrawString('Validation', $fontLabel, $brushDark, $validationX, {top_margin + plot_height + 48}, $centerFormat)
$graphics.DrawString('Test', $fontLabel, $brushDark, $testX, {top_margin + plot_height + 48}, $centerFormat)
"""

    for model, label, color in COST_SERIES:
        validation_value = float(row_for(model, "Validation")["mean_challenge_cost"])
        test_value = float(row_for(model, "Test")["mean_challenge_cost"])
        ps_script += f"""
$color_{slugify(label)} = [System.Drawing.ColorTranslator]::FromHtml('{color}')
$pen_{slugify(label)} = New-Object System.Drawing.Pen($color_{slugify(label)}, 3)
$brush_{slugify(label)} = New-Object System.Drawing.SolidBrush($color_{slugify(label)})
$valY_{slugify(label)} = $topMargin + $plotHeight - ((({validation_value} - $minCost) / ($maxCost - $minCost)) * $plotHeight)
$testY_{slugify(label)} = $topMargin + $plotHeight - ((({test_value} - $minCost) / ($maxCost - $minCost)) * $plotHeight)
$graphics.DrawLine($pen_{slugify(label)}, $validationX, $valY_{slugify(label)}, $testX, $testY_{slugify(label)})
$graphics.FillEllipse($brush_{slugify(label)}, $validationX - 7, $valY_{slugify(label)} - 7, 14, 14)
$graphics.FillEllipse($brush_{slugify(label)}, $testX - 7, $testY_{slugify(label)} - 7, 14, 14)
$graphics.DrawEllipse([System.Drawing.Pens]::White, $validationX - 7, $valY_{slugify(label)} - 7, 14, 14)
$graphics.DrawEllipse([System.Drawing.Pens]::White, $testX - 7, $testY_{slugify(label)} - 7, 14, 14)
$graphics.DrawString('{validation_value:.3f}', $fontValue, $brushDark, $validationX, $valY_{slugify(label)} - 18, $centerFormat)
$graphics.DrawString('{test_value:.3f}', $fontValue, $brushDark, $testX, $testY_{slugify(label)} - 18, $centerFormat)
$pen_{slugify(label)}.Dispose()
$brush_{slugify(label)}.Dispose()
"""

    legend_y = height - 40
    legend_start_x = (width - (len(COST_SERIES) * 280)) / 2
    ps_script += f"\n$legendY = {legend_y}\n$legendStartX = {legend_start_x}\n"
    for index, (_, label, color) in enumerate(COST_SERIES):
        ps_script += f"""
$legendX{index} = $legendStartX + ({index} * 280)
$legendColor{index} = [System.Drawing.ColorTranslator]::FromHtml('{color}')
$legendPen{index} = New-Object System.Drawing.Pen($legendColor{index}, 3)
$legendBrush{index} = New-Object System.Drawing.SolidBrush($legendColor{index})
$graphics.DrawLine($legendPen{index}, $legendX{index}, $legendY, $legendX{index} + 48, $legendY)
$graphics.FillEllipse($legendBrush{index}, $legendX{index} + 17, $legendY - 6.5, 13, 13)
$graphics.DrawEllipse([System.Drawing.Pens]::White, $legendX{index} + 17, $legendY - 6.5, 13, 13)
$graphics.DrawString('{label}', $fontLegend, $brushDark, $legendX{index} + 60, $legendY + 6, (New-Object System.Drawing.StringFormat))
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
$fontValue.Dispose()
$fontLegend.Dispose()
$graphics.Dispose()
$bmp.Dispose()
"""

    subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_script],
        check=True,
        cwd=str(PROJECT_ROOT),
    )


def write_paper_style_score_chart_svg(output_path: Path, split: str) -> None:
    width = 980
    height = 620
    left_margin = 110
    right_margin = 45
    top_margin = 52
    bottom_margin = 140
    plot_width = width - left_margin - right_margin
    plot_height = height - top_margin - bottom_margin
    x_step = plot_width / (len(SCORE_METRICS) - 1)
    tick_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    series = [
        ("Reactive OBD-II-style baseline", "Reactive baseline", "#5b9bd5"),
        ("Two-stage CatBoost", "Two-stage CatBoost", "#ed7d31"),
    ]

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-size="25" font-family="Calibri, Arial, sans-serif" font-weight="700" fill="#1f2933">{split} Score Comparison</text>',
        f'<line x1="{left_margin}" y1="{top_margin + plot_height}" x2="{width - right_margin}" y2="{top_margin + plot_height}" stroke="#bdd7ee" stroke-width="1.5"/>',
        f'<line x1="{left_margin}" y1="{top_margin}" x2="{left_margin}" y2="{top_margin + plot_height}" stroke="#bdd7ee" stroke-width="1.5"/>',
        f'<text x="{left_margin - 76}" y="{top_margin + plot_height / 2}" text-anchor="middle" transform="rotate(-90 {left_margin - 76},{top_margin + plot_height / 2})" font-size="19" font-family="Calibri, Arial, sans-serif" fill="#1f2933">Score Value</text>',
        f'<text x="{width / 2}" y="{height - 58}" text-anchor="middle" font-size="19" font-family="Calibri, Arial, sans-serif" fill="#1f2933">Evaluation Metric</text>',
    ]

    for tick_value in tick_values:
        y = top_margin + plot_height - (tick_value * plot_height)
        svg_lines.append(
            f'<line x1="{left_margin}" y1="{y}" x2="{width - right_margin}" y2="{y}" stroke="#9dc3e6" stroke-width="1.2"/>'
        )
        svg_lines.append(
            f'<text x="{left_margin - 18}" y="{y + 6}" text-anchor="end" font-size="16" font-family="Calibri, Arial, sans-serif" fill="#1f2933">{tick_value:.1f}</text>'
        )

    for index, (_, metric_label) in enumerate(SCORE_METRICS):
        x = left_margin + (index * x_step)
        svg_lines.append(
            f'<text x="{x}" y="{top_margin + plot_height + 36}" text-anchor="middle" font-size="15" font-family="Calibri, Arial, sans-serif" fill="#1f2933">{metric_label}</text>'
        )

    for model, label, color in series:
        row = row_for(model, split)
        points: list[str] = []
        for index, (metric_key, _) in enumerate(SCORE_METRICS):
            x = left_margin + (index * x_step)
            y = top_margin + plot_height - (float(row[metric_key]) * plot_height)
            points.append(f"{x},{y}")
        svg_lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3.2" points="{" ".join(points)}"/>'
        )
        for index, (metric_key, _) in enumerate(SCORE_METRICS):
            x = left_margin + (index * x_step)
            y = top_margin + plot_height - (float(row[metric_key]) * plot_height)
            svg_lines.append(
                f'<circle cx="{x}" cy="{y}" r="5.5" fill="{color}"/>'
            )

    legend_y = height - 26
    legend_start_x = (width - 420) / 2
    for index, (_, label, color) in enumerate(series):
        item_x = legend_start_x + (index * 220)
        svg_lines.extend(
            [
                f'<line x1="{item_x}" y1="{legend_y}" x2="{item_x + 36}" y2="{legend_y}" stroke="{color}" stroke-width="3.2"/>',
                f'<circle cx="{item_x + 18}" cy="{legend_y}" r="5" fill="{color}"/>',
                f'<text x="{item_x + 46}" y="{legend_y + 6}" font-size="15" font-family="Calibri, Arial, sans-serif" fill="#1f2933">{label}</text>',
            ]
        )

    svg_lines.append("</svg>")
    output_path.write_text("\n".join(svg_lines), encoding="utf-8")


def write_paper_style_score_chart_png(output_path: Path, split: str) -> None:
    width = 980
    height = 620
    left_margin = 110
    right_margin = 45
    top_margin = 52
    bottom_margin = 140
    plot_width = width - left_margin - right_margin
    plot_height = height - top_margin - bottom_margin
    x_step = plot_width / (len(SCORE_METRICS) - 1)
    series = [
        ("Reactive OBD-II-style baseline", "Reactive baseline", "#5b9bd5"),
        ("Two-stage CatBoost", "Two-stage CatBoost", "#ed7d31"),
    ]

    ps_script = f"""
Add-Type -AssemblyName System.Drawing
$bmp = New-Object System.Drawing.Bitmap {width}, {height}
$graphics = [System.Drawing.Graphics]::FromImage($bmp)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
$graphics.Clear([System.Drawing.Color]::White)

$fontTitle = New-Object System.Drawing.Font('Calibri', 22, [System.Drawing.FontStyle]::Bold)
$fontAxis = New-Object System.Drawing.Font('Calibri', 16)
$fontTick = New-Object System.Drawing.Font('Calibri', 14)
$fontLabel = New-Object System.Drawing.Font('Calibri', 13)
$fontLegend = New-Object System.Drawing.Font('Calibri', 13)

$brushDark = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(31,41,51))
$penAxis = New-Object System.Drawing.Pen([System.Drawing.ColorTranslator]::FromHtml('#bdd7ee'), 1.5)
$penGrid = New-Object System.Drawing.Pen([System.Drawing.ColorTranslator]::FromHtml('#9dc3e6'), 1.2)

$leftMargin = {left_margin}
$rightMargin = {right_margin}
$topMargin = {top_margin}
$plotHeight = {plot_height}
$plotWidth = {plot_width}
$baselineY = $topMargin + $plotHeight
$xStep = {x_step}

$centerFormat = New-Object System.Drawing.StringFormat
$centerFormat.Alignment = [System.Drawing.StringAlignment]::Center
$centerFormat.LineAlignment = [System.Drawing.StringAlignment]::Center

$graphics.DrawString('{split} Score Comparison', $fontTitle, $brushDark, {width / 2}, 28, $centerFormat)
$graphics.DrawLine($penAxis, $leftMargin, $baselineY, {width - right_margin}, $baselineY)
$graphics.DrawLine($penAxis, $leftMargin, $topMargin, $leftMargin, $baselineY)

$graphics.TranslateTransform({left_margin - 76}, {top_margin + plot_height / 2})
$graphics.RotateTransform(-90)
$graphics.DrawString('Score Value', $fontAxis, $brushDark, 0, 0, $centerFormat)
$graphics.ResetTransform()
$graphics.DrawString('Evaluation Metric', $fontAxis, $brushDark, {width / 2}, {height - 58}, $centerFormat)

$tickValues = @(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
for($i = 0; $i -lt $tickValues.Count; $i++) {{
    $tickValue = [double]$tickValues[$i]
    $y = $topMargin + $plotHeight - ($tickValue * $plotHeight)
    $graphics.DrawLine($penGrid, $leftMargin, $y, {width - right_margin}, $y)
    $graphics.DrawString(($tickValue.ToString('0.0')), $fontTick, $brushDark, $leftMargin - 18, $y + 4, (New-Object System.Drawing.StringFormat))
}}
"""

    for index, (_, metric_label) in enumerate(SCORE_METRICS):
        x = left_margin + (index * x_step)
        ps_script += (
            f"\n$graphics.DrawString('{metric_label}', $fontLabel, $brushDark, {x}, "
            f"{top_margin + plot_height + 36}, $centerFormat)\n"
        )

    for model, label, color in series:
        row = row_for(model, split)
        values_str = ", ".join(f"{float(row[metric_key]):.6f}" for metric_key, _ in SCORE_METRICS)
        ps_script += f"""
function Draw-Paper-Series-{slugify(label)}([double[]]$values) {{
    $color = [System.Drawing.ColorTranslator]::FromHtml('{color}')
    $pen = New-Object System.Drawing.Pen($color, 3.2)
    $brush = New-Object System.Drawing.SolidBrush($color)
    $points = New-Object 'System.Drawing.PointF[]' $values.Length
    for($j = 0; $j -lt $values.Length; $j++) {{
        $x = [float]($leftMargin + ($j * $xStep))
        $y = [float]($topMargin + $plotHeight - ($values[$j] * $plotHeight))
        $points[$j] = New-Object System.Drawing.PointF($x, $y)
    }}
    if($points.Length -gt 1) {{ $graphics.DrawLines($pen, $points) }}
    foreach($point in $points) {{
        $graphics.FillEllipse($brush, $point.X - 5.5, $point.Y - 5.5, 11, 11)
    }}
    $pen.Dispose()
    $brush.Dispose()
}}
Draw-Paper-Series-{slugify(label)} -values @({values_str})
"""

    legend_y = height - 26
    legend_start_x = (width - 420) / 2
    ps_script += f"\n$legendY = {legend_y}\n$legendStartX = {legend_start_x}\n"
    for index, (_, label, color) in enumerate(series):
        ps_script += f"""
$legendX{index} = $legendStartX + ({index} * 220)
$legendColor{index} = [System.Drawing.ColorTranslator]::FromHtml('{color}')
$legendPen{index} = New-Object System.Drawing.Pen($legendColor{index}, 3.2)
$legendBrush{index} = New-Object System.Drawing.SolidBrush($legendColor{index})
$graphics.DrawLine($legendPen{index}, $legendX{index}, $legendY, $legendX{index} + 36, $legendY)
$graphics.FillEllipse($legendBrush{index}, $legendX{index} + 13, $legendY - 5, 10, 10)
$graphics.DrawString('{label}', $fontLegend, $brushDark, $legendX{index} + 46, $legendY + 4, (New-Object System.Drawing.StringFormat))
$legendPen{index}.Dispose()
$legendBrush{index}.Dispose()
"""

    ps_script += f"""
$bmp.Save('{str(output_path)}', [System.Drawing.Imaging.ImageFormat]::Png)
$penAxis.Dispose()
$penGrid.Dispose()
$brushDark.Dispose()
$fontTitle.Dispose()
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


def slugify(text: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in text).strip("_")


if __name__ == "__main__":
    main()
