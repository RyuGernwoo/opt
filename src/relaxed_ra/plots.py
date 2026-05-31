from __future__ import annotations

from collections import defaultdict
from pathlib import Path


def _scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if abs(src_max - src_min) < 1e-12:
        return (dst_min + dst_max) / 2.0
    return dst_min + (value - src_min) * (dst_max - dst_min) / (src_max - src_min)


def write_line_plot(
    rows: list[dict[str, str]],
    output_path: Path,
    title: str,
    x_key: str,
    y_key: str,
    series_key: str = "method",
    x_label: str = "",
    y_label: str = "",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        try:
            x_val = float(row[x_key])
            y_val = float(row[y_key])
        except (KeyError, ValueError):
            continue
        grouped[row[series_key]].append((x_val, y_val))

    width, height = 920, 560
    left, right, top, bottom = 90, 250, 55, 85
    plot_w = width - left - right
    plot_h = height - top - bottom
    palette = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#17becf",
        "#4d4d4d",
    ]

    points = [point for series in grouped.values() for point in series]
    if points:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if y_min > 0:
            y_min = 0.0
        y_pad = max((y_max - y_min) * 0.08, 1e-9)
        y_max += y_pad
    else:
        x_min, x_max, y_min, y_max = 0.0, 1.0, 0.0, 1.0

    def sx(x_val: float) -> float:
        return _scale(x_val, x_min, x_max, left, left + plot_w)

    def sy(y_val: float) -> float:
        return _scale(y_val, y_min, y_max, top + plot_h, top)

    elements: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="30" font-family="Arial, sans-serif" font-size="22" font-weight="700">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222" stroke-width="1.2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222" stroke-width="1.2"/>',
    ]

    for index in range(6):
        frac = index / 5.0
        y_val = y_min + frac * (y_max - y_min)
        y = sy(y_val)
        elements.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e5e5"/>')
        elements.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12">{y_val:.3g}</text>')

    unique_xs = sorted({point[0] for point in points})
    for x_val in unique_xs:
        x = sx(x_val)
        elements.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 5}" stroke="#222"/>')
        elements.append(f'<text x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12">{x_val:g}</text>')

    for idx, (series, series_points) in enumerate(sorted(grouped.items())):
        color = palette[idx % len(palette)]
        ordered = sorted(series_points)
        if len(ordered) >= 2:
            coords = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in ordered)
            elements.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        for x_val, y_val in ordered:
            elements.append(f'<circle cx="{sx(x_val):.2f}" cy="{sy(y_val):.2f}" r="3.5" fill="{color}"/>')
        legend_y = top + 24 + idx * 22
        elements.append(f'<line x1="{left + plot_w + 35}" y1="{legend_y}" x2="{left + plot_w + 58}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        elements.append(f'<text x="{left + plot_w + 66}" y="{legend_y + 4}" font-family="Arial, sans-serif" font-size="12">{series}</text>')

    elements.append(f'<text x="{left + plot_w / 2}" y="{height - 28}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">{x_label}</text>')
    elements.append(
        f'<text transform="translate(24 {top + plot_h / 2}) rotate(-90)" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">{y_label}</text>'
    )
    elements.append("</svg>")
    output_path.write_text("\n".join(elements), encoding="utf-8")
