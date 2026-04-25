from __future__ import annotations

import base64
import colorsys
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


def hex_to_chroma(hex_color: str) -> float:
    r, g, b = hex_to_rgb(hex_color)
    rgb = np.array([[[r / 255, g / 255, b / 255]]])
    lab = color.rgb2lab(rgb)
    _, a, b_val = lab[0][0]
    return float((a**2 + b_val**2) ** 0.5)


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
    """Guarda un histograma de chroma Lab per veure colors grisos vs saturats."""
    required_columns = {"chroma"}
    missing_columns = required_columns - set(sample.columns)
    if missing_columns:
        raise ValueError(f"Falten columnes per generar el grafic: {sorted(missing_columns)}")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    values = sample["chroma"].astype(float).to_numpy()
    max_chroma = max(1.0, float(np.ceil(values.max() / 10) * 10))
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

    draw.text((left, 18), "Distribucio del chroma de la mostra", fill=(0, 0, 0))
    draw.text((left, 38), "Chroma baix = mes grisos; chroma alt = colors mes saturats", fill=(70, 70, 70))
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
    draw.text((right - 34, bottom + 8), f"{max_chroma:.0f}", fill=(90, 90, 90))
    draw.text((left + (plot_width // 2) - 28, bottom + 30), "chroma", fill=(0, 0, 0))
    draw.text((left - 48, top - 8), str(max_count), fill=(90, 90, 90))
    draw.text((left - 44, top + 18), "freq.", fill=(0, 0, 0))

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


def compact_error_message(value: str, max_length: int = 180) -> str:
    """Redueix errors llargs a una sola linia perque el log sigui llegible."""
    message = " ".join(str(value or "").split())
    if len(message) <= max_length:
        return message
    return f"{message[: max_length - 3]}..."


def query_model_for_color(client, image_path: str | Path, model: str, temperature: float = 0.2) -> str:
    image = encode_image_base64(image_path)
    response = client.responses.create(
        model=model,
        temperature=temperature,
        max_output_tokens=16,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT_FIX},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{image}"},
                ],
            }
        ],
    )
    return response.output_text.strip()


def collect_model_outputs(
    client,
    image_paths: list[Path],
    models: list[str],
    temperature: float = 0.2,
    output_path: str | Path | None = None,
    log_file: str | Path | None = None,
    max_images: int | None = None,
    retry_failed: bool = True,
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
                raw_response = query_model_for_color(
                    client,
                    image_path=image_path,
                    model=model,
                    temperature=temperature,
                )
                response_hex = normalise_hex_response(raw_response)
                if response_hex is None:
                    status = "invalid_hex"
                    error_message = f"Resposta sense hexadecimal valid: {raw_response}"
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


def build_final_sample(input_sample: pd.DataFrame, model_outputs: pd.DataFrame) -> pd.DataFrame:
    final = model_outputs.merge(input_sample, on="image_name", how="left")
    final["error_cromatic"] = final.apply(
        lambda row: rgb_distance(row["hex"], row["response_hex"])
        if pd.notna(row["hex"]) and pd.notna(row["response_hex"])
        else np.nan,
        axis=1,
    )
    return final
