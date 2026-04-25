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


def generate_unique_colors(n: int = 1000, seed: int = 23) -> pd.DataFrame:
    random.seed(seed)
    colors: list[dict] = []
    generated: set[str] = set()

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

    return pd.DataFrame(colors)


def generate_near_color(rgb: tuple[int, int, int], distance: int = 30) -> tuple[int, int, int]:
    def clamp(value: int) -> int:
        return max(0, min(255, value))

    return tuple(clamp(channel + random.randint(-distance, distance)) for channel in rgb)


def generate_image_array(
    target_rgb: tuple[int, int, int],
    size: int = 100,
    near_distance: int = 30,
) -> np.ndarray:
    total_pixels = size * size
    target_pixels = int(0.6 * total_pixels)
    near_pixels = int(0.2 * total_pixels)
    random_pixels = total_pixels - target_pixels - near_pixels

    near_color = generate_near_color(target_rgb, distance=near_distance)
    random_color = (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
    )

    pixels = (
        [target_rgb] * target_pixels
        + [near_color] * near_pixels
        + [random_color] * random_pixels
    )
    random.shuffle(pixels)
    return np.array(pixels, dtype=np.uint8).reshape((size, size, 3))


def save_color_image(image_array: np.ndarray, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_array).save(output_path, format="PNG")
    return output_path


def save_rgb_sample_map(
    sample: pd.DataFrame,
    path: str | Path,
    columns: int = 40,
    swatch_size: int = 18,
) -> Path:
    """Guarda un mapa visual amb tots els colors RGB de la mostra."""
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


def generate_images_from_sample(
    sample: pd.DataFrame,
    output_dir: str | Path,
    size: int = 100,
    near_distance: int = 30,
    seed: int = 23,
) -> list[Path]:
    random.seed(seed)
    image_dir = Path(output_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for row in sample.itertuples(index=False):
        target_rgb = (int(row.r), int(row.g), int(row.b))
        image_array = generate_image_array(target_rgb, size=size, near_distance=near_distance)
        paths.append(save_color_image(image_array, image_dir / row.image_name))

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


def query_model_for_color(client, image_path: str | Path, model: str, temperature: float = 0.2) -> str:
    image = encode_image_base64(image_path)
    response = client.responses.create(
        model=model,
        temperature=temperature,
        max_output_tokens=10,
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
) -> pd.DataFrame:
    rows: list[dict] = []

    for image_path in sorted(image_paths):
        for model in models:
            started = time.time()
            raw_response = query_model_for_color(
                client,
                image_path=image_path,
                model=model,
                temperature=temperature,
            )
            elapsed_seconds = time.time() - started
            rows.append(
                {
                    "image_name": image_path.name,
                    "model": model,
                    "temperature": temperature,
                    "response_raw": raw_response,
                    "response_hex": normalise_hex_response(raw_response),
                    "elapsed_seconds": elapsed_seconds,
                    "executed_at": datetime.now().isoformat(timespec="seconds"),
                }
            )

    return pd.DataFrame(rows)


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
