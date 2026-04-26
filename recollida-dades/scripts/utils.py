from __future__ import annotations

import base64
import colorsys
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageDraw
from skimage import color


PROMPT_FIX = (
    "Identify the main color of this image, understanding main as the one "
    "that takes up the most space in the image. Return only a hexadecimal RGB "
    "code (RRGGBB)."
)

PROMPT_RGB_JSON = (
    "Estimate the dominant RGB color of the image as precisely as possible.\n"
    "Do not use a named color, web color, CSS color, or common palette approximation.\n"
    "Return the closest numeric estimate of the dominant color, even if the values are not round numbers.\n"
    "Return only valid JSON with exactly these integer fields from 0 to 255:\n"
    '{"r": 148, "g": 42, "b": 8}\n'
    "Do not include markdown, explanations, hexadecimal codes, or any extra text."
)

SRGB_MAX_LAB_CHROMA = 133.8041534423828


def write_log(message: str, log_file: str | Path = Path("logs") / "pipeline.log") -> str:
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}"
    with log_path.open("a", encoding="utf-8") as file:
        file.write(line + "\n")

    return line


def format_duration(seconds: float) -> str:
    """Formata segons com a text curt per logs i sortides del notebook."""
    seconds = max(0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remaining_seconds:04.1f}s"

    hours, remaining_minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(remaining_minutes)}m {remaining_seconds:04.1f}s"


class ProgressBar:
    """Barra de progres simple per notebooks, amb ETA i temps transcorregut."""

    def __init__(self, total: int, label: str, width: int = 28) -> None:
        self.total = max(1, int(total))
        self.label = label
        self.width = width
        self.started = time.time()
        self.current = 0

    def update(self, current: int | None = None, extra: str = "") -> None:
        if current is None:
            self.current += 1
        else:
            self.current = current

        elapsed = time.time() - self.started
        rate = self.current / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.current) / rate if rate > 0 else 0
        ratio = min(1, self.current / self.total)
        done = int(self.width * ratio)
        bar = "#" * done + "-" * (self.width - done)
        suffix = f" | {extra}" if extra else ""

        print(
            f"\r{self.label} [{bar}] {self.current}/{self.total} "
            f"({ratio * 100:5.1f}%) elapsed={format_duration(elapsed)} eta={format_duration(remaining)}{suffix}",
            end="",
            flush=True,
        )

    def finish(self, extra: str = "") -> None:
        self.update(self.total, extra=extra)
        print()


def delete_sample_outputs(
    csv_dir: str | Path,
    images_dir: str | Path,
    logs_dir: str | Path,
) -> list[Path]:
    """Esborra els fitxers generats de la mostra, pero conserva les carpetes."""
    removed: list[Path] = []
    csv_path = Path(csv_dir)
    image_path = Path(images_dir)
    log_path = Path(logs_dir)

    for pattern in ["input_image_sample.csv", "outputmodel_image_sample.csv", "sample-colors.csv"]:
        target = csv_path / pattern
        if target.exists():
            target.unlink()
            removed.append(target)

    if image_path.exists():
        for target in image_path.glob("*.png"):
            target.unlink()
            removed.append(target)

    if log_path.exists():
        for target in log_path.glob("*.log"):
            target.unlink()
            removed.append(target)

    for folder in [csv_path, image_path, log_path]:
        folder.mkdir(parents=True, exist_ok=True)

    return removed


def delete_png_files(image_dir: str | Path) -> list[Path]:
    """Esborra nomes els PNG d'una carpeta i conserva la carpeta."""
    folder = Path(image_dir)
    removed: list[Path] = []

    if folder.exists():
        for target in folder.glob("*.png"):
            target.unlink()
            removed.append(target)

    folder.mkdir(parents=True, exist_ok=True)
    return removed


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    clean = hex_color.strip().lstrip("#").upper()
    if not re.fullmatch(r"[0-9A-F]{6}", clean):
        raise ValueError(f"Color hexadecimal no valido: {hex_color}")

    return int(clean[0:2], 16), int(clean[2:4], 16), int(clean[4:6], 16)


def hex_to_lab_chroma(hex_color: str) -> float:
    r, g, b = hex_to_rgb(hex_color)
    rgb = np.array([[[r / 255, g / 255, b / 255]]])
    lab = color.rgb2lab(rgb)
    _, a, b_val = lab[0][0]
    return float((a**2 + b_val**2) ** 0.5)


def hex_to_chroma(hex_color: str) -> float:
    """Retorna el chroma Lab normalitzat entre 0 i 1 dins l'espai sRGB."""
    return min(1.0, hex_to_lab_chroma(hex_color) / SRGB_MAX_LAB_CHROMA)


def generate_unique_colors(
    n: int = 1000,
    seed: int = 23,
    progress: bool = False,
    log_file: str | Path | None = None,
) -> pd.DataFrame:
    random.seed(seed)
    colors: list[dict] = []
    generated: set[str] = set()
    progress_bar = ProgressBar(n, "Mostra RGB") if progress else None
    started = time.time()

    while len(generated) < n:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        hex_color = rgb_to_hex((r, g, b))

        if hex_color in generated:
            continue

        generated.add(hex_color)
        colors.append(
            {
                "image_name": f"{hex_color}.png",
                "hex": hex_color,
                "r": r,
                "g": g,
                "b": b,
                "chroma": hex_to_chroma(hex_color),
            }
        )

        if progress_bar:
            progress_bar.update(len(generated), extra=hex_color)

    if progress_bar:
        progress_bar.finish(extra="fet")

    if log_file:
        write_log(f"Mostra RGB generada: files={len(colors)} temps={format_duration(time.time() - started)}", log_file)

    return pd.DataFrame(colors)


def generate_near_color(rgb: tuple[int, int, int], distance: int = 30) -> tuple[int, int, int]:
    def clamp(value: int) -> int:
        return max(0, min(255, value))

    return tuple(clamp(channel + random.randint(-distance, distance)) for channel in rgb)


def generate_random_rgb() -> tuple[int, int, int]:
    return (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
    )


def generate_image_array(
    target_rgb: tuple[int, int, int],
    size: int = 100,
    near_distance: int = 30,
) -> np.ndarray:
    total_pixels = size * size
    target_pixels = int(0.6 * total_pixels)
    near_pixels = int(0.2 * total_pixels)
    random_pixels = total_pixels - target_pixels - near_pixels

    pixels = (
        [target_rgb] * target_pixels
        + [generate_near_color(target_rgb, distance=near_distance) for _ in range(near_pixels)]
        + [generate_random_rgb() for _ in range(random_pixels)]
    )
    random.shuffle(pixels)
    return np.array(pixels, dtype=np.uint8).reshape((size, size, 3))


def save_color_image(image_array: np.ndarray, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_array).save(output_path, format="PNG")
    return output_path


def save_rgb_sample_grid(
    sample: pd.DataFrame,
    path: str | Path,
    columns: int = 40,
    swatch_size: int = 18,
) -> Path:
    """Guarda una graella amb tots els colors RGB exactes de la mostra."""
    required_columns = {"r", "g", "b"}
    missing_columns = required_columns - set(sample.columns)
    if missing_columns:
        raise ValueError(f"Falten columnes per generar el mapa: {sorted(missing_columns)}")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colors = []
    for row in sample.itertuples(index=False):
        rgb = (int(row.r), int(row.g), int(row.b))
        hsv = colorsys.rgb_to_hsv(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
        colors.append((hsv, rgb))

    colors.sort(key=lambda item: (item[0][0], item[0][1], item[0][2]))

    rows = int(np.ceil(len(colors) / columns))
    canvas = Image.new("RGB", (columns * swatch_size, rows * swatch_size), "white")

    for index, (_, rgb) in enumerate(colors):
        x = (index % columns) * swatch_size
        y = (index // columns) * swatch_size
        swatch = Image.new("RGB", (swatch_size, swatch_size), rgb)
        canvas.paste(swatch, (x, y))

    canvas.save(output_path, format="PNG")
    return output_path


def save_rgb_sample_map(
    sample: pd.DataFrame,
    path: str | Path,
    panel_size: int = 300,
    margin: int = 44,
    point_radius: int = 2,
) -> Path:
    """Guarda diagnostics visuals per veure si la mostra RGB esta ben repartida."""
    required_columns = {"r", "g", "b"}
    missing_columns = required_columns - set(sample.columns)
    if missing_columns:
        raise ValueError(f"Falten columnes per generar el mapa: {sorted(missing_columns)}")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panel_width = panel_size + (2 * margin)
    panel_height = panel_size + (2 * margin)
    title_height = 34
    gutter = 22
    canvas_width = (2 * panel_width) + gutter
    canvas_height = title_height + (3 * panel_height) + (2 * gutter)
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    rows = list(sample[["r", "g", "b"]].itertuples(index=False, name=None))

    def point_color(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
        return tuple(int(v) for v in rgb)

    def panel_origin(col: int, row: int) -> tuple[int, int]:
        return col * (panel_width + gutter), title_height + row * (panel_height + gutter)

    def draw_axes(origin: tuple[int, int], title: str, x_label: str, y_label: str) -> tuple[int, int]:
        ox, oy = origin
        left = ox + margin
        top = oy + margin
        right = left + panel_size
        bottom = top + panel_size
        draw.rectangle([left, top, right, bottom], outline=(25, 25, 25), width=1)
        draw.text((left, oy + 8), title, fill=(0, 0, 0))
        draw.text((left + 60, top - 18), y_label, fill=(0, 0, 0))
        draw.text((left + 60, bottom + 12), x_label, fill=(0, 0, 0))
        draw.text((left - 8, bottom + 2), "0", fill=(90, 90, 90))
        draw.text((right - 22, bottom + 2), "255", fill=(90, 90, 90))
        draw.text((left - 30, top - 6), "255", fill=(90, 90, 90))
        return left, top

    def draw_scatter(origin: tuple[int, int], title: str, x_label: str, y_label: str, x_index: int, y_index: int) -> None:
        left, top = draw_axes(origin, title, x_label, y_label)
        for rgb in rows:
            x = left + round((rgb[x_index] / 255) * panel_size)
            y = top + panel_size - round((rgb[y_index] / 255) * panel_size)
            color_rgb = point_color(rgb)
            draw.ellipse(
                [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
                fill=color_rgb,
            )

    def draw_scatter_values(
        origin: tuple[int, int],
        title: str,
        x_label: str,
        y_label: str,
        points: list[tuple[int, int, tuple[int, int, int]]],
    ) -> None:
        left, top = draw_axes(origin, title, x_label, y_label)
        for x_value, y_value, rgb in points:
            x = left + round((x_value / 255) * panel_size)
            y = top + panel_size - round((y_value / 255) * panel_size)
            draw.ellipse(
                [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
                fill=point_color(rgb),
            )

    def draw_histogram(origin: tuple[int, int]) -> None:
        left, top = draw_axes(origin, "Distribucio per canal", "valor RGB", "freq.")
        bins = np.linspace(0, 256, 17)
        max_count = 1
        histograms = []
        for channel in ["r", "g", "b"]:
            counts, _ = np.histogram(sample[channel], bins=bins)
            histograms.append((channel, counts))
            max_count = max(max_count, int(counts.max()))

        bar_width = panel_size / 16
        channel_colors = {"r": (220, 40, 40), "g": (40, 170, 70), "b": (50, 90, 220)}
        offsets = {"r": -bar_width / 4, "g": 0, "b": bar_width / 4}
        for channel, counts in histograms:
            for i, count in enumerate(counts):
                x_center = left + (i + 0.5) * bar_width + offsets[channel]
                bar_h = (count / max_count) * panel_size
                draw.rectangle(
                    [
                        x_center - (bar_width / 8),
                        top + panel_size - bar_h,
                        x_center + (bar_width / 8),
                        top + panel_size,
                    ],
                    fill=channel_colors[channel],
                )

    def draw_hue_saturation(origin: tuple[int, int]) -> None:
        left, top = draw_axes(origin, "Hue vs saturation", "hue", "saturation")
        for y in range(panel_size):
            saturation = 1 - (y / max(1, panel_size - 1))
            for x in range(panel_size):
                hue = x / max(1, panel_size - 1)
                rgb_float = colorsys.hsv_to_rgb(hue, saturation, 1)
                canvas.putpixel((left + x, top + y), tuple(int(v * 255) for v in rgb_float))
        draw.rectangle([left, top, left + panel_size, top + panel_size], outline=(25, 25, 25), width=1)
        draw.rectangle([left - 38, top - 24, left + 34, top + 4], fill="white")
        draw.rectangle([left + panel_size - 28, top + panel_size + 2, left + panel_size + 34, top + panel_size + 18], fill="white")
        draw.text((left - 8, top + panel_size + 2), "0", fill=(90, 90, 90))
        draw.text((left + panel_size - 8, top + panel_size + 2), "1", fill=(90, 90, 90))
        draw.text((left - 18, top - 6), "1", fill=(90, 90, 90))

        for rgb in rows:
            hue, saturation, _ = colorsys.rgb_to_hsv(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
            x = left + round(hue * panel_size)
            y = top + panel_size - round(saturation * panel_size)
            draw.ellipse(
                [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
                fill=point_color(rgb),
            )

    draw.text((margin, 8), "Diagnosi visual de distribucio de les 1000 mostres RGB", fill=(0, 0, 0))
    draw_histogram(panel_origin(0, 0))
    draw_hue_saturation(panel_origin(1, 0))
    draw_scatter(panel_origin(0, 1), "R vs G", "R", "G", 0, 1)
    draw_scatter(panel_origin(1, 1), "R vs B", "R", "B", 0, 2)
    draw_scatter(panel_origin(0, 2), "G vs B", "G", "B", 1, 2)
    value_points = [(max(rgb), min(rgb), rgb) for rgb in rows]
    draw_scatter_values(panel_origin(1, 2), "max RGB vs min RGB", "max", "min", value_points)
    canvas.save(output_path, format="PNG")
    return output_path


def save_chroma_distribution(
    sample: pd.DataFrame,
    path: str | Path,
    width: int = 760,
    height: int = 420,
    bins_count: int = 24,
) -> Path:
    """Guarda un histograma de chroma Lab normalitzat per veure grisos vs saturats."""
    required_columns = {"chroma"}
    missing_columns = required_columns - set(sample.columns)
    if missing_columns:
        raise ValueError(f"Falten columnes per generar el grafic: {sorted(missing_columns)}")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    values = sample["chroma"].astype(float).to_numpy()
    max_chroma = 1.0
    counts, edges = np.histogram(values, bins=np.linspace(0, max_chroma, bins_count + 1))
    max_count = max(1, int(counts.max()))

    margin_left = 64
    margin_right = 28
    margin_top = 64
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    left = margin_left
    top = margin_top
    right = left + plot_width
    bottom = top + plot_height

    draw.text((left, 18), "Distribucio del chroma normalitzat de la mostra", fill=(0, 0, 0))
    draw.text((left, 38), "0 = mes grisos; 1 = maxim chroma possible en sRGB", fill=(70, 70, 70))
    draw.rectangle([left, top, right, bottom], outline=(25, 25, 25), width=1)

    bar_gap = 2
    bar_width = plot_width / bins_count
    for index, count in enumerate(counts):
        x0 = left + index * bar_width + bar_gap
        x1 = left + (index + 1) * bar_width - bar_gap
        bar_height = (count / max_count) * plot_height
        y0 = bottom - bar_height
        intensity = int(90 + 130 * (index / max(1, bins_count - 1)))
        fill = (intensity, 80, 190)
        draw.rectangle([x0, y0, x1, bottom], fill=fill)

    draw.text((left - 8, bottom + 8), "0", fill=(90, 90, 90))
    draw.text((right - 18, bottom + 8), "1", fill=(90, 90, 90))
    draw.text((left + (plot_width // 2) - 28, bottom + 30), "chroma", fill=(0, 0, 0))
    draw.text((left - 48, top - 8), str(max_count), fill=(90, 90, 90))
    draw.text((left - 44, top + 18), "freq.", fill=(0, 0, 0))

    canvas.save(output_path, format="PNG")
    return output_path


def save_error_distribution(
    final_sample: pd.DataFrame,
    path: str | Path,
    width: int = 820,
    height: int = 460,
    bins_count: int = 24,
) -> Path:
    """Guarda un histograma de l'error cromatic RGB separat per model."""
    required_columns = {"model", "status", "error_cromatic"}
    missing_columns = required_columns - set(final_sample.columns)
    if missing_columns:
        raise ValueError(f"Falten columnes per generar el grafic: {sorted(missing_columns)}")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    valid = final_sample[
        (final_sample["status"] == "ok") & final_sample["error_cromatic"].notna()
    ].copy()

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    margin_left = 72
    margin_right = 32
    margin_top = 82
    margin_bottom = 78
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    left = margin_left
    top = margin_top
    right = left + plot_width
    bottom = top + plot_height

    draw.text((left, 20), "Distribucio de l'error cromatic per model", fill=(0, 0, 0))
    draw.text(
        (left, 42),
        "Error cromatic = distancia euclidiana RGB entre color real i resposta",
        fill=(70, 70, 70),
    )
    draw.rectangle([left, top, right, bottom], outline=(25, 25, 25), width=1)

    if valid.empty:
        draw.text((left + 24, top + 32), "Encara no hi ha prediccions valides per dibuixar.", fill=(120, 0, 0))
        canvas.save(output_path, format="PNG")
        return output_path

    max_error = max(1.0, float(np.ceil(valid["error_cromatic"].max() / 10) * 10))
    edges = np.linspace(0, max_error, bins_count + 1)
    models = sorted(valid["model"].dropna().unique())
    palette = [
        (45, 105, 190),
        (220, 90, 45),
        (70, 160, 95),
        (155, 90, 185),
    ]

    histograms = []
    max_count = 1
    for model in models:
        counts, _ = np.histogram(valid.loc[valid["model"] == model, "error_cromatic"], bins=edges)
        histograms.append((model, counts))
        max_count = max(max_count, int(counts.max()))

    group_width = plot_width / bins_count
    bar_gap = 2
    bar_width = max(2, (group_width - (len(models) + 1) * bar_gap) / max(1, len(models)))

    for model_index, (model, counts) in enumerate(histograms):
        fill = palette[model_index % len(palette)]
        for bin_index, count in enumerate(counts):
            x0 = left + bin_index * group_width + bar_gap + model_index * (bar_width + bar_gap)
            x1 = x0 + bar_width
            bar_height = (count / max_count) * plot_height
            y0 = bottom - bar_height
            draw.rectangle([x0, y0, x1, bottom], fill=fill)

    draw.text((left - 8, bottom + 8), "0", fill=(90, 90, 90))
    draw.text((right - 44, bottom + 8), f"{max_error:.0f}", fill=(90, 90, 90))
    draw.text((left + (plot_width // 2) - 60, bottom + 32), "error cromatic", fill=(0, 0, 0))
    draw.text((left - 54, top - 8), str(max_count), fill=(90, 90, 90))
    draw.text((left - 50, top + 18), "freq.", fill=(0, 0, 0))

    legend_x = left
    legend_y = height - 28
    for model_index, model in enumerate(models):
        fill = palette[model_index % len(palette)]
        x = legend_x + model_index * 150
        draw.rectangle([x, legend_y, x + 14, legend_y + 14], fill=fill)
        draw.text((x + 20, legend_y - 1), str(model), fill=(0, 0, 0))

    canvas.save(output_path, format="PNG")
    return output_path


def save_error_boxplot(
    final_sample: pd.DataFrame,
    path: str | Path,
    width: int = 820,
    height: int = 460,
) -> Path:
    """Guarda un boxplot simple de l'error cromatic per model."""
    required_columns = {"model", "status", "error_cromatic"}
    missing_columns = required_columns - set(final_sample.columns)
    if missing_columns:
        raise ValueError(f"Falten columnes per generar el boxplot: {sorted(missing_columns)}")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid = final_sample[
        (final_sample["status"] == "ok") & final_sample["error_cromatic"].notna()
    ].copy()

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    margin_left = 72
    margin_right = 36
    margin_top = 82
    margin_bottom = 82
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    left = margin_left
    top = margin_top
    bottom = top + plot_height
    right = left + plot_width

    draw.text((left, 20), "Boxplot de l'error cromatic per model", fill=(0, 0, 0))
    draw.text((left, 42), "Linia central = mediana; caixa = IQR; bigotis = valors no extrems", fill=(70, 70, 70))
    draw.rectangle([left, top, right, bottom], outline=(25, 25, 25), width=1)

    if valid.empty:
        draw.text((left + 24, top + 32), "Encara no hi ha prediccions valides per dibuixar.", fill=(120, 0, 0))
        canvas.save(output_path, format="PNG")
        return output_path

    max_error = max(1.0, float(np.ceil(valid["error_cromatic"].max() / 10) * 10))
    models = sorted(valid["model"].dropna().unique())
    palette = [(45, 105, 190), (220, 90, 45), (70, 160, 95), (155, 90, 185)]

    def y_for(value: float) -> float:
        return bottom - (value / max_error) * plot_height

    draw.text((left - 8, bottom + 8), "0", fill=(90, 90, 90))
    draw.text((left - 52, top - 8), f"{max_error:.0f}", fill=(90, 90, 90))
    draw.text((left - 54, top + 18), "error", fill=(0, 0, 0))

    spacing = plot_width / max(1, len(models))
    box_width = min(90, spacing * 0.45)

    for model_index, model in enumerate(models):
        values = valid.loc[valid["model"] == model, "error_cromatic"].astype(float).to_numpy()
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        iqr = q3 - q1
        low = np.min(values[values >= q1 - 1.5 * iqr])
        high = np.max(values[values <= q3 + 1.5 * iqr])
        x = left + spacing * (model_index + 0.5)
        fill = palette[model_index % len(palette)]

        y_q1 = y_for(q1)
        y_q3 = y_for(q3)
        y_median = y_for(median)
        y_low = y_for(low)
        y_high = y_for(high)

        draw.line([x, y_high, x, y_low], fill=(40, 40, 40), width=2)
        draw.line([x - box_width / 4, y_high, x + box_width / 4, y_high], fill=(40, 40, 40), width=2)
        draw.line([x - box_width / 4, y_low, x + box_width / 4, y_low], fill=(40, 40, 40), width=2)
        draw.rectangle([x - box_width / 2, y_q3, x + box_width / 2, y_q1], outline=(40, 40, 40), fill=fill, width=2)
        draw.line([x - box_width / 2, y_median, x + box_width / 2, y_median], fill=(255, 255, 255), width=3)
        draw.text((x - 46, bottom + 18), str(model), fill=(0, 0, 0))

    canvas.save(output_path, format="PNG")
    return output_path


def save_response_repetition_plot(
    final_sample: pd.DataFrame,
    path: str | Path,
    top_n: int = 12,
    width: int = 920,
    height: int = 540,
) -> Path:
    """Guarda barres amb les respostes hex repetides mes frequents per model."""
    required_columns = {"model", "status", "response_hex"}
    missing_columns = required_columns - set(final_sample.columns)
    if missing_columns:
        raise ValueError(f"Falten columnes per generar el grafic: {sorted(missing_columns)}")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid = final_sample[
        (final_sample["status"] == "ok") & final_sample["response_hex"].notna()
    ].copy()

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((48, 20), "Respostes RGB mes repetides", fill=(0, 0, 0))
    draw.text((48, 42), "Si apareixen moltes repeticions, el model esta col·lapsant cap a una paleta discreta", fill=(70, 70, 70))

    if valid.empty:
        draw.text((72, 92), "Encara no hi ha prediccions valides per dibuixar.", fill=(120, 0, 0))
        canvas.save(output_path, format="PNG")
        return output_path

    models = sorted(valid["model"].dropna().unique())
    panel_width = (width - 96) / max(1, len(models))
    panel_top = 88
    panel_height = height - panel_top - 40
    max_count = 1
    model_counts: dict[str, pd.Series] = {}

    for model in models:
        counts = valid.loc[valid["model"] == model, "response_hex"].value_counts().head(top_n)
        model_counts[str(model)] = counts
        if len(counts) > 0:
            max_count = max(max_count, int(counts.max()))

    for model_index, model in enumerate(models):
        counts = model_counts[str(model)]
        x0 = 48 + model_index * panel_width
        y0 = panel_top
        draw.text((x0, y0 - 22), str(model), fill=(0, 0, 0))
        bar_area_width = panel_width - 120
        row_height = panel_height / max(1, top_n)

        for row_index, (hex_color, count) in enumerate(counts.items()):
            y = y0 + row_index * row_height
            try:
                fill = hex_to_rgb(str(hex_color))
            except ValueError:
                fill = (120, 120, 120)
            bar_width = (count / max_count) * bar_area_width
            draw.rectangle([x0 + 74, y + 3, x0 + 74 + bar_width, y + row_height - 4], fill=fill, outline=(40, 40, 40))
            draw.text((x0, y + 4), str(hex_color), fill=(0, 0, 0))
            draw.text((x0 + 80 + bar_width, y + 4), str(int(count)), fill=(0, 0, 0))

    canvas.save(output_path, format="PNG")
    return output_path


def save_error_vs_chroma_plot(
    final_sample: pd.DataFrame,
    path: str | Path,
    width: int = 820,
    height: int = 460,
) -> Path:
    """Guarda un scatter plot d'error cromatic contra chroma per model."""
    required_columns = {"model", "status", "error_cromatic", "chroma"}
    missing_columns = required_columns - set(final_sample.columns)
    if missing_columns:
        raise ValueError(f"Falten columnes per generar el grafic: {sorted(missing_columns)}")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid = final_sample[
        (final_sample["status"] == "ok") & final_sample["error_cromatic"].notna() & final_sample["chroma"].notna()
    ].copy()

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    margin_left = 72
    margin_right = 36
    margin_top = 82
    margin_bottom = 78
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    left = margin_left
    top = margin_top
    right = left + plot_width
    bottom = top + plot_height

    draw.text((left, 20), "Error cromatic vs chroma", fill=(0, 0, 0))
    draw.text((left, 42), "Cada punt es una imatge del pilot", fill=(70, 70, 70))
    draw.rectangle([left, top, right, bottom], outline=(25, 25, 25), width=1)

    if valid.empty:
        draw.text((left + 24, top + 32), "Encara no hi ha prediccions valides per dibuixar.", fill=(120, 0, 0))
        canvas.save(output_path, format="PNG")
        return output_path

    max_error = max(1.0, float(np.ceil(valid["error_cromatic"].max() / 10) * 10))
    palette = [(45, 105, 190), (220, 90, 45), (70, 160, 95), (155, 90, 185)]

    draw.text((left - 8, bottom + 8), "0", fill=(90, 90, 90))
    draw.text((right - 20, bottom + 8), "1", fill=(90, 90, 90))
    draw.text((left + (plot_width // 2) - 26, bottom + 32), "chroma", fill=(0, 0, 0))
    draw.text((left - 52, top - 8), f"{max_error:.0f}", fill=(90, 90, 90))
    draw.text((left - 54, top + 18), "error", fill=(0, 0, 0))

    for model_index, model in enumerate(sorted(valid["model"].dropna().unique())):
        sub = valid[valid["model"] == model]
        fill = palette[model_index % len(palette)]
        for row in sub.itertuples(index=False):
            x = left + float(row.chroma) * plot_width
            y = bottom - (float(row.error_cromatic) / max_error) * plot_height
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=fill)
        legend_x = left + model_index * 160
        legend_y = height - 28
        draw.rectangle([legend_x, legend_y, legend_x + 14, legend_y + 14], fill=fill)
        draw.text((legend_x + 20, legend_y - 1), str(model), fill=(0, 0, 0))

    canvas.save(output_path, format="PNG")
    return output_path


def generate_images_from_sample(
    sample: pd.DataFrame,
    output_dir: str | Path,
    size: int = 100,
    near_distance: int = 30,
    seed: int = 23,
    progress: bool = False,
    log_file: str | Path | None = None,
) -> list[Path]:
    random.seed(seed)
    image_dir = Path(output_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    progress_bar = ProgressBar(len(sample), "Imatges PNG") if progress else None
    started = time.time()

    for index, row in enumerate(sample.itertuples(index=False), start=1):
        target_rgb = (int(row.r), int(row.g), int(row.b))
        image_array = generate_image_array(target_rgb, size=size, near_distance=near_distance)
        paths.append(save_color_image(image_array, image_dir / row.image_name))

        if progress_bar:
            progress_bar.update(index, extra=row.image_name)

    if progress_bar:
        progress_bar.finish(extra="fet")

    if log_file:
        write_log(f"Imatges generades: files={len(paths)} temps={format_duration(time.time() - started)}", log_file)

    return paths


def generate_solid_images_from_sample(
    sample: pd.DataFrame,
    output_dir: str | Path,
    size: int = 100,
    progress: bool = False,
    log_file: str | Path | None = None,
) -> list[Path]:
    """Genera imatges 100% del color objectiu, sense pixels propers ni aleatoris."""
    image_dir = Path(output_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    progress_bar = ProgressBar(len(sample), "Imatges sense soroll") if progress else None
    started = time.time()

    for index, row in enumerate(sample.itertuples(index=False), start=1):
        target_rgb = (int(row.r), int(row.g), int(row.b))
        image_array = np.full((size, size, 3), target_rgb, dtype=np.uint8)
        paths.append(save_color_image(image_array, image_dir / row.image_name))

        if progress_bar:
            progress_bar.update(index, extra=row.image_name)

    if progress_bar:
        progress_bar.finish(extra="fet")

    if log_file:
        write_log(
            f"Imatges sense soroll generades: files={len(paths)} temps={format_duration(time.time() - started)}",
            log_file,
        )

    return paths


def save_csv(data: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return output_path


def encode_image_base64(path: str | Path) -> str:
    with Path(path).open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def normalise_hex_response(value: str) -> str | None:
    match = re.search(r"#?([0-9A-Fa-f]{6})", value or "")
    return match.group(1).upper() if match else None


def normalise_rgb_json_response(value: str) -> str | None:
    """Extreu RGB JSON i el retorna com hexadecimal RRGGBB."""
    text = str(value or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    channels: list[int] = []
    for key in ["r", "g", "b"]:
        if key not in payload:
            return None
        try:
            value_int = int(round(float(payload[key])))
        except (TypeError, ValueError):
            return None
        if value_int < 0 or value_int > 255:
            return None
        channels.append(value_int)

    return rgb_to_hex(tuple(channels))


def compact_error_message(value: str, max_length: int = 180) -> str:
    """Redueix errors llargs a una sola linia perque el log sigui llegible."""
    message = " ".join(str(value or "").split())
    if len(message) <= max_length:
        return message
    return f"{message[: max_length - 3]}..."


def query_model_for_color(
    client,
    image_path: str | Path,
    model: str,
    temperature: float | None = 0.2,
    prompt: str = PROMPT_FIX,
    max_output_tokens: int = 16,
    reasoning_effort: str | None = None,
) -> str:
    image = encode_image_base64(image_path)
    request = {
        "model": model,
        "max_output_tokens": max_output_tokens,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{image}"},
                ],
            }
        ],
    }
    if temperature is not None:
        request["temperature"] = temperature
    if reasoning_effort is not None and str(model).startswith("gpt-5"):
        request["reasoning"] = {"effort": reasoning_effort}

    response = client.responses.create(**request)
    return response.output_text.strip()


def select_pilot_image_paths(
    sample: pd.DataFrame,
    images_dir: str | Path,
    n_images: int = 50,
    seed: int = 23,
    previous_results: pd.DataFrame | None = None,
    hard_fraction: float = 0.4,
) -> list[Path]:
    """Selecciona una mostra pilot equilibrada per chroma i amb casos dificils si existeixen resultats previs."""
    if n_images <= 0:
        return []

    required_columns = {"image_name", "chroma"}
    missing_columns = required_columns - set(sample.columns)
    if missing_columns:
        raise ValueError(f"Falten columnes per seleccionar pilot: {sorted(missing_columns)}")

    rng = np.random.default_rng(seed)
    image_dir = Path(images_dir)
    selected: list[str] = []

    if previous_results is not None and {"image_name", "error_cromatic"}.issubset(previous_results.columns):
        hard_n = min(n_images, max(0, int(round(n_images * hard_fraction))))
        hard_images = (
            previous_results.dropna(subset=["error_cromatic"])
            .groupby("image_name", as_index=False)["error_cromatic"]
            .max()
            .sort_values("error_cromatic", ascending=False)
            .head(hard_n)["image_name"]
            .tolist()
        )
        selected.extend(hard_images)

    remaining_n = n_images - len(dict.fromkeys(selected))
    if remaining_n > 0:
        base = sample[~sample["image_name"].isin(selected)].copy()
        base["chroma_bin"] = pd.qcut(base["chroma"], q=min(4, len(base)), duplicates="drop")
        bins = list(base.groupby("chroma_bin", observed=True))
        per_bin = max(1, int(np.ceil(remaining_n / max(1, len(bins)))))
        balanced: list[str] = []

        for _, group in bins:
            names = group["image_name"].to_numpy()
            take = min(per_bin, len(names), remaining_n - len(balanced))
            if take <= 0:
                break
            balanced.extend(rng.choice(names, size=take, replace=False).tolist())

        if len(balanced) < remaining_n:
            rest = base[~base["image_name"].isin(balanced)]["image_name"].to_numpy()
            take = min(remaining_n - len(balanced), len(rest))
            if take > 0:
                balanced.extend(rng.choice(rest, size=take, replace=False).tolist())

        selected.extend(balanced)

    unique_selected = list(dict.fromkeys(selected))[:n_images]
    return [image_dir / name for name in unique_selected if (image_dir / name).exists()]


def collect_model_outputs(
    client,
    image_paths: list[Path],
    models: list[str],
    temperature: float | None = 0.2,
    output_path: str | Path | None = None,
    log_file: str | Path | None = None,
    max_images: int | None = None,
    retry_failed: bool = True,
    response_format: str = "hex",
    prompt: str | None = None,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = None,
) -> pd.DataFrame:
    """Consulta models de visio i desa resultats de forma incremental."""
    rows: list[dict] = []
    output_file = Path(output_path) if output_path else None
    existing = pd.DataFrame()
    done_pairs: set[tuple[str, str]] = set()

    if output_file and output_file.exists():
        existing = pd.read_csv(output_file)
        if retry_failed and "status" in existing.columns:
            retry_count = int((existing["status"] != "ok").sum())
            if retry_count and log_file:
                write_log(f"Models pendents de reintentar: {retry_count} files amb status != ok", log_file)
            existing = existing[existing["status"] == "ok"].copy()
        if {"image_name", "model"}.issubset(existing.columns):
            done_pairs = set(zip(existing["image_name"], existing["model"]))

    selected_paths = sorted(image_paths)
    if max_images is not None:
        selected_paths = selected_paths[:max_images]

    total_tasks = len(selected_paths) * len(models)
    task_index = 0
    progress_bar = ProgressBar(total_tasks, "Models API")
    full_started = time.time()

    for image_path in selected_paths:
        for model in models:
            task_index += 1
            if (image_path.name, model) in done_pairs:
                if log_file:
                    write_log(f"Model skip [{task_index}/{total_tasks}] {model} {image_path.name}", log_file)
                progress_bar.update(task_index, extra=f"skip {model} {image_path.name}")
                continue

            started = time.time()
            if log_file:
                write_log(f"Model start [{task_index}/{total_tasks}] {model} {image_path.name}", log_file)

            raw_response = ""
            response_hex = None
            status = "ok"
            error_message = ""
            try:
                query_prompt = prompt or (PROMPT_RGB_JSON if response_format == "rgb_json" else PROMPT_FIX)
                token_limit = max_output_tokens
                if token_limit is None:
                    token_limit = 1200 if response_format == "rgb_json" else 16
                raw_response = query_model_for_color(
                    client,
                    image_path=image_path,
                    model=model,
                    temperature=temperature,
                    prompt=query_prompt,
                    max_output_tokens=token_limit,
                    reasoning_effort=reasoning_effort,
                )
                if response_format == "rgb_json":
                    response_hex = normalise_rgb_json_response(raw_response)
                else:
                    response_hex = normalise_hex_response(raw_response)
                if response_hex is None and response_format == "rgb_json":
                    retry_token_limit = max(token_limit * 2, 2400)
                    if log_file:
                        write_log(
                            "Model retry per JSON incomplet "
                            f"[{task_index}/{total_tasks}] {model} {image_path.name} "
                            f"max_output_tokens={retry_token_limit}",
                            log_file,
                        )
                    raw_response = query_model_for_color(
                        client,
                        image_path=image_path,
                        model=model,
                        temperature=temperature,
                        prompt=query_prompt,
                        max_output_tokens=retry_token_limit,
                        reasoning_effort=reasoning_effort,
                    )
                    response_hex = normalise_rgb_json_response(raw_response)
                if response_hex is None:
                    status = "invalid_hex"
                    error_message = f"Resposta sense color valid: {raw_response}"
            except Exception as error:  # noqa: BLE001
                status = "error"
                error_message = str(error)

            elapsed_seconds = time.time() - started
            row = {
                "image_name": image_path.name,
                "model": model,
                "temperature": temperature,
                "status": status,
                "response_raw": raw_response,
                "response_hex": response_hex,
                "error_message": error_message,
                "elapsed_seconds": elapsed_seconds,
                "executed_at": datetime.now().isoformat(timespec="seconds"),
            }
            rows.append(row)

            if output_file:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                pd.concat([existing, pd.DataFrame(rows)], ignore_index=True).to_csv(output_file, index=False)

            if log_file:
                error_for_log = compact_error_message(error_message)
                detail = f" error={error_for_log}" if error_for_log else ""
                write_log(
                    f"Model done [{task_index}/{total_tasks}] {model} {image_path.name} "
                    f"status={status} hex={response_hex} seconds={elapsed_seconds:.2f}{detail}",
                    log_file,
                )

            progress_bar.update(task_index, extra=f"{status} {model} {image_path.name}")

    progress_bar.finish(extra=f"temps total {format_duration(time.time() - full_started)}")

    return pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)


def rgb_distance(hex_a: str, hex_b: str) -> float:
    rgb_a = np.array(hex_to_rgb(hex_a))
    rgb_b = np.array(hex_to_rgb(hex_b))
    return float(np.linalg.norm(rgb_a - rgb_b))


def build_final_sample(
    input_sample: pd.DataFrame,
    model_outputs: pd.DataFrame,
    images_dir: str | Path | None = None,
) -> pd.DataFrame:
    input_sample = input_sample.copy()
    if "hex" in input_sample.columns:
        input_sample["chroma"] = input_sample["hex"].apply(hex_to_chroma)

    final = input_sample.merge(model_outputs, on="image_name", how="left")

    if images_dir is not None:
        image_dir = Path(images_dir)
        final["image_kb"] = final["image_name"].apply(
            lambda name: round((image_dir / str(name)).stat().st_size / 1024, 2)
            if (image_dir / str(name)).exists()
            else np.nan
        )
    else:
        final["image_kb"] = np.nan

    final["response_hex"] = final["response_hex"].apply(
        lambda value: normalise_hex_response(str(value)) if pd.notna(value) else np.nan
    )

    valid_response = (final["status"] == "ok") & final["response_hex"].notna()
    final["response_r"] = np.nan
    final["response_g"] = np.nan
    final["response_b"] = np.nan

    for index, response_hex in final.loc[valid_response, "response_hex"].items():
        r, g, b = hex_to_rgb(response_hex)
        final.loc[index, ["response_r", "response_g", "response_b"]] = [r, g, b]

    for column in ["response_r", "response_g", "response_b"]:
        final[column] = final[column].astype("Int64")

    final["error_cromatic"] = np.nan
    final.loc[valid_response, "error_cromatic"] = final.loc[valid_response].apply(
        lambda row: rgb_distance(row["hex"], row["response_hex"]),
        axis=1,
    )

    ordered_columns = [
        "image_name",
        "image_kb",
        "hex",
        "r",
        "g",
        "b",
        "chroma",
        "model",
        "temperature",
        "status",
        "response_raw",
        "response_hex",
        "response_r",
        "response_g",
        "response_b",
        "error_cromatic",
        "elapsed_seconds",
        "executed_at",
        "error_message",
    ]
    for column in ordered_columns:
        if column not in final.columns:
            final[column] = np.nan

    return final[ordered_columns]
