# coding: utf-8
import io
import math
import hashlib
import sqlite3
from typing import cast

import numpy as np
import cv2
from PIL import Image


def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg

def generate_tile_coords(bounds, minzoom, maxzoom):
    def deg2num(lat, lon, zoom):
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        xtile = int((lon + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
        return xtile, ytile
    coords = []
    for z in range(minzoom, maxzoom + 1):
        x0, y0 = deg2num(bounds[3], bounds[0], z)
        x1, y1 = deg2num(bounds[1], bounds[2], z)
        for x in range(min(x0, x1), max(x0, x1) + 1):
            for y in range(min(y0, y1), max(y0, y1) + 1):
                coords.append((z, x, y))
    return coords

# --- Helpers ---
def geo_to_pixel(lon: float, lat: float, bounds, width: int, height: int) -> tuple[float, float]:
    """Проецирует WGS84 координаты в пиксели уже выровненного изображения (без округления)."""
    min_lon, min_lat, max_lon, max_lat = bounds
    if max_lon == min_lon or max_lat == min_lat:
        raise ValueError('Некорректные bounds: деление на ноль')
    px = (lon - min_lon) / (max_lon - min_lon) * width
    py = (max_lat - lat) / (max_lat - min_lat) * height
    return px, py


# --- Export function for import into main pipeline ---
def export_geowarp_mbtiles(
    mosaic: Image.Image | None,
    bounds,
    zoom: int,
    mbtiles_path: str,
    tile_size: int = 256,
    metadata: dict | None = None,
    deduplicate: bool = True,
    image_format: str = 'png',
    min_zoom: int | None = None,
    max_zoom: int | None = None,
    projective_source: dict | None = None,
) -> None:
    """
    Экспортирует мозаику в MBTiles с обратной геопривязкой каждого тайла.
    :param mosaic: PIL.Image (RGBA)
    :param bounds: (minLon, minLat, maxLon, maxLat)
    :param zoom: int
    :param mbtiles_path: str
    :param tile_size: int
    :param metadata: dict (дополнительные метаданные MBTiles)
    :param deduplicate: bool (включить dedup тайлов по hash)
    """
    image_format = (image_format or 'png').lower()
    src_array_proj: np.ndarray | None = None
    matrix_inv_proj: np.ndarray | None = None
    border_proj: tuple[float, ...] | None = None

    if projective_source is not None:
        projective_mode = True
        mosaic_image: Image.Image | None = None
        width, height = projective_source['dest_size']
        src_array_proj = cast(np.ndarray, projective_source['src_array'])
        matrix_inv_proj = cast(np.ndarray, projective_source['matrix_inv'])
        if src_array_proj.ndim != 3 or src_array_proj.shape[2] not in (3, 4):
            raise ValueError('projective src_array must be HxWx3 или HxWx4')
        if src_array_proj.shape[2] == 3:
            alpha_channel = np.full(src_array_proj.shape[:2] + (1,), 255, dtype=src_array_proj.dtype)
            src_array_proj = np.concatenate([src_array_proj, alpha_channel], axis=2)
        projective_border = projective_source.get('border_value', (0, 0, 0, 0))
        if len(projective_border) != src_array_proj.shape[2]:
            projective_border = tuple(list(projective_border) + [0] * (src_array_proj.shape[2] - len(projective_border)))
        border_proj = tuple(float(v) for v in projective_border)
        warp_flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    else:
        projective_mode = False
        if mosaic is None:
            raise ValueError('mosaic image is required when projective_source is not provided')
        mosaic_image = mosaic if mosaic.mode == 'RGBA' else mosaic.convert('RGBA')
        width, height = mosaic_image.size
        warp_flags = cv2.INTER_LINEAR
    resolved_min_zoom = zoom if min_zoom is None else min_zoom
    resolved_max_zoom = zoom if max_zoom is None else max_zoom
    if resolved_min_zoom > resolved_max_zoom:
        raise ValueError('min_zoom не может быть больше max_zoom')
    pil_format = image_format.upper()
    all_tile_coords = generate_tile_coords(bounds, resolved_min_zoom, resolved_max_zoom)
    total_coords = len(all_tile_coords)
    spinner = ['|', '/', '-', '\\']
    try:
        resample_mode = Image.Resampling.BICUBIC  # type: ignore[attr-defined]
    except AttributeError:
        resample_mode = getattr(Image, 'BICUBIC', 3)
    conn = sqlite3.connect(mbtiles_path)
    try:
        conn.execute('PRAGMA journal_mode = MEMORY')
        conn.execute('PRAGMA synchronous = OFF')
    except sqlite3.DatabaseError:
        pass
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS tiles (zoom_level INTEGER, tile_column INTEGER, tile_row INTEGER, tile_data BLOB)')
    c.execute('CREATE TABLE IF NOT EXISTS metadata (name TEXT, value TEXT)')
    c.execute('CREATE UNIQUE INDEX IF NOT EXISTS tile_index on tiles (zoom_level, tile_column, tile_row)')
    conn.commit()
    tile_hashes: dict[bytes, bytes] = {}
    total_tiles = unique_tiles = duplicate_tiles = 0
    tile_buffer = io.BytesIO()
    batch: list[tuple[int, int, int, bytes]] = []
    batch_size = 128
    try:
        conn.execute('BEGIN')
        for i, (z, x, y) in enumerate(all_tile_coords, start=1):
            spin = spinner[(i - 1) % len(spinner)] if spinner else ''
            if total_coords:
                print(f"\rЗапись тайлов ({i}/{total_coords}) {spin}", end='')
            lat0, lon0 = num2deg(x, y, z)
            lat1, lon1 = num2deg(x + 1, y + 1, z)
            px0, py0 = geo_to_pixel(lon0, lat0, bounds, width, height)
            px1, py1 = geo_to_pixel(lon1, lat1, bounds, width, height)
            left = max(0.0, min(px0, px1))
            right = min(float(width), max(px0, px1))
            top = max(0.0, min(py0, py1))
            bottom = min(float(height), max(py0, py1))
            if right - left <= 1e-6 or bottom - top <= 1e-6:
                continue
            if projective_mode:
                scale_x = (right - left) / tile_size
                scale_y = (bottom - top) / tile_size
                if scale_x <= 0 or scale_y <= 0:
                    continue
                scale_matrix = np.array([
                    [scale_x, 0.0, left],
                    [0.0, scale_y, top],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float64)
                assert src_array_proj is not None and matrix_inv_proj is not None and border_proj is not None
                warp_matrix = matrix_inv_proj @ scale_matrix
                tile_np = cv2.warpPerspective(
                    src_array_proj,
                    warp_matrix.astype(np.float64),
                    dsize=(tile_size, tile_size),
                    flags=warp_flags,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=border_proj,
                )
                if tile_np.ndim != 3 or tile_np.shape[2] != 4:
                    continue
                if not np.any(tile_np[..., 3]):
                    continue
                tile_img = Image.fromarray(tile_np, mode='RGBA')
            else:
                assert mosaic_image is not None
                tile_img = mosaic_image.crop((left, top, right, bottom)).resize((tile_size, tile_size), resample_mode)
                if tile_img.getbbox() is None:
                    continue
            if tile_img.getbbox() is None:
                continue
            tile_img.save(tile_buffer, format=pil_format)
            tile_bytes = tile_buffer.getvalue()
            tile_buffer.seek(0)
            tile_buffer.truncate(0)
            tile_hash = hashlib.blake2b(tile_bytes, digest_size=16).digest() if deduplicate else None
            if deduplicate and tile_hash in tile_hashes:
                duplicate_tiles += 1
                tile_bytes = tile_hashes[tile_hash]
            else:
                if deduplicate and tile_hash is not None:
                    tile_hashes[tile_hash] = tile_bytes
                unique_tiles += 1
            tms_y = (2 ** z - 1) - y
            batch.append((z, x, tms_y, tile_bytes))
            if len(batch) >= batch_size:
                c.executemany(
                    'INSERT OR REPLACE INTO tiles (zoom_level, tile_column, tile_row, tile_data) VALUES (?, ?, ?, ?)',
                    batch,
                )
                batch.clear()
            total_tiles += 1

        if batch:
            c.executemany(
                'INSERT OR REPLACE INTO tiles (zoom_level, tile_column, tile_row, tile_data) VALUES (?, ?, ?, ?)',
                batch,
            )
            batch.clear()

        if total_coords:
            print()

        meta = {
            'format': image_format,
            'type': 'overlay',
            'minzoom': str(resolved_min_zoom),
            'maxzoom': str(resolved_max_zoom),
            'bounds': ','.join(map(str, bounds)),
        }
        if metadata:
            meta.update(metadata)
        meta['minzoom'] = str(resolved_min_zoom)
        meta['maxzoom'] = str(resolved_max_zoom)
        for k, v in meta.items():
            c.execute('INSERT OR REPLACE INTO metadata (name, value) VALUES (?, ?)', (k, v))
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()
    finally:
        conn.close()


def build_lower_zoom_pyramid(
    mbtiles_path: str,
    min_zoom: int,
    max_zoom: int,
    image_format: str = 'png',
    tile_size: int = 256,
    deduplicate: bool = True,
) -> None:
    """Формирует недостающие уровни зума, объединяя дочерние тайлы более высокого уровня."""
    if min_zoom >= max_zoom:
        # Нижних уровней нет, только одно значение
        conn = sqlite3.connect(mbtiles_path)
        try:
            conn.execute('INSERT OR REPLACE INTO metadata (name, value) VALUES (?, ?)', ('minzoom', str(max_zoom)))
            conn.execute('INSERT OR REPLACE INTO metadata (name, value) VALUES (?, ?)', ('maxzoom', str(max_zoom)))
            conn.commit()
        finally:
            conn.close()
        return

    pil_format = (image_format or 'png').upper()
    save_params: dict[str, object]
    if pil_format == 'WEBP':
        save_params = {'lossless': True, 'quality': 100}
    else:
        save_params = {}
    try:
        resample_down = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
    except AttributeError:
        resample_down = getattr(Image, 'LANCZOS', getattr(Image, 'BICUBIC', 3))

    conn = sqlite3.connect(mbtiles_path)
    buffer = io.BytesIO()
    tile_hashes = {} if deduplicate else None

    try:
        cursor = conn.cursor()
        for parent_zoom in range(max_zoom - 1, min_zoom - 1, -1):
            child_zoom = parent_zoom + 1
            cursor.execute(
                'SELECT tile_column, tile_row, tile_data FROM tiles WHERE zoom_level=?',
                (child_zoom,),
            )
            rows = cursor.fetchall()
            if not rows:
                continue

            grouped: dict[tuple[int, int], list[bytes | None]] = {}
            row_mask = (2 ** child_zoom) - 1
            parent_row_mask = (2 ** parent_zoom) - 1
            for col, row_tms, data in rows:
                child_y_xyz = row_mask - row_tms
                parent_x = col // 2
                parent_y_xyz = child_y_xyz // 2
                parent_y_tms = parent_row_mask - parent_y_xyz
                parent_key = (parent_x, parent_y_tms)
                quadrant = (child_y_xyz % 2) * 2 + (col % 2)
                slots = grouped.setdefault(parent_key, [None, None, None, None])
                slots[quadrant] = data

            insert_batch: list[tuple[int, int, int, bytes]] = []
            total_groups = len(grouped)
            processed = 0
            for (parent_x, parent_y_tms), quadrants in grouped.items():
                composite = Image.new('RGBA', (tile_size * 2, tile_size * 2), (0, 0, 0, 0))
                has_content = False

                for q_idx, q_data in enumerate(quadrants):
                    if not q_data:
                        continue
                    try:
                        with Image.open(io.BytesIO(q_data)) as child_img:
                            child_rgba = child_img.convert('RGBA')
                    except Exception:
                        continue

                    if child_rgba.getbbox() is not None:
                        has_content = True

                    if child_rgba.size != (tile_size, tile_size):
                        child_rgba = child_rgba.resize((tile_size, tile_size), resample=resample_down)

                    dx = q_idx % 2
                    dy = q_idx // 2
                    composite.paste(child_rgba, (dx * tile_size, dy * tile_size))

                if not has_content:
                    continue

                parent_img = composite.resize((tile_size, tile_size), resample=resample_down)
                if parent_img.getbbox() is None:
                    continue

                parent_img.save(buffer, format=pil_format, **save_params)
                tile_bytes = buffer.getvalue()
                buffer.seek(0)
                buffer.truncate(0)

                if deduplicate and tile_hashes is not None:
                    tile_hash = hashlib.blake2b(tile_bytes, digest_size=16).digest()
                    cached = tile_hashes.get(tile_hash)
                    if cached is not None:
                        tile_bytes = cached
                    else:
                        tile_hashes[tile_hash] = tile_bytes

                insert_batch.append((parent_zoom, parent_x, parent_y_tms, tile_bytes))
                processed += 1
                if processed % 200 == 0:
                    print(f"  Зум {parent_zoom}: {processed}/{total_groups} родительских тайлов")

            if insert_batch:
                cursor.executemany(
                    'INSERT OR REPLACE INTO tiles (zoom_level, tile_column, tile_row, tile_data) VALUES (?, ?, ?, ?)',
                    insert_batch,
                )
                conn.commit()
                print(f"Сформировано тайлов уровня {parent_zoom}: {len(insert_batch)}")

        cursor.execute('INSERT OR REPLACE INTO metadata (name, value) VALUES (?, ?)', ('minzoom', str(min_zoom)))
        cursor.execute('INSERT OR REPLACE INTO metadata (name, value) VALUES (?, ?)', ('maxzoom', str(max_zoom)))
        conn.commit()
    finally:
        conn.close()
