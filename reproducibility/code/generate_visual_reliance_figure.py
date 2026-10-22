"""Rebuild an editable SVG from the archived primary-effect table (stdlib only)."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = []
with (ROOT / "statistics" / "paired_effects.csv").open(newline="", encoding="utf-8-sig") as handle:
    for row in csv.DictReader(handle):
        if row["contrast"] == "correct - no_visual" and row["model"] in {"early", "coattention", "medgemma"}:
            rows.append(row)

width, height = 900, 560
left, right, top, bottom = 95, 30, 35, 85
xmin, xmax, ymin, ymax = 15.0, 65.0, -10.0, 55.0

def sx(value):
    return left + (float(value) - xmin) * (width - left - right) / (xmax - xmin)

def sy(value):
    return top + (ymax - float(value)) * (height - top - bottom) / (ymax - ymin)

labels = {"early": "Early fusion", "coattention": "Co-attention", "medgemma": "MedGemma-4B"}
colors = {"slake": "#0072B2", "vqarad": "#D55E00"}
parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
    '<rect width="100%" height="100%" fill="white"/>',
    '<style>text{font-family:Arial,sans-serif;fill:#202020}.tick{font-size:13px}.label{font-size:16px}.point{font-size:12px}</style>',
]
for value in range(-10, 56, 10):
    y = sy(value)
    parts += [f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e3e3e3"/>', f'<text class="tick" x="{left-10}" y="{y+5:.1f}" text-anchor="end">{value}</text>']
for value in range(20, 66, 10):
    x = sx(value)
    parts += [f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="#eeeeee"/>', f'<text class="tick" x="{x:.1f}" y="{height-bottom+24}" text-anchor="middle">{value}</text>']
parts += [f'<line x1="{left}" y1="{sy(0):.1f}" x2="{width-right}" y2="{sy(0):.1f}" stroke="#555" stroke-dasharray="6 5"/>', f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222"/>', f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222"/>']
for row in rows:
    x, y = sx(row["change_rate_pct"]), sy(row["utility_pp"])
    low, high, color = sy(row["ci_low_pp"]), sy(row["ci_high_pp"]), colors[row["dataset"]]
    parts += [f'<line x1="{x:.1f}" y1="{low:.1f}" x2="{x:.1f}" y2="{high:.1f}" stroke="{color}" stroke-width="2"/>', f'<line x1="{x-5:.1f}" y1="{low:.1f}" x2="{x+5:.1f}" y2="{low:.1f}" stroke="{color}" stroke-width="2"/>', f'<line x1="{x-5:.1f}" y1="{high:.1f}" x2="{x+5:.1f}" y2="{high:.1f}" stroke="{color}" stroke-width="2"/>', f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}"/>', f'<text class="point" x="{x+8:.1f}" y="{y-7:.1f}">{labels[row["model"]]} ({row["dataset"].upper()})</text>']
parts += [f'<text class="label" x="{(left+width-right)/2:.1f}" y="{height-22}" text-anchor="middle">Prediction-change rate after visual removal (%)</text>', f'<text class="label" transform="translate(24 {(top+height-bottom)/2:.1f}) rotate(-90)" text-anchor="middle">Visual utility (percentage points)</text>', '</svg>']
(ROOT / "figure" / "visual_reliance_rebuilt.svg").write_text("\n".join(parts), encoding="utf-8")

