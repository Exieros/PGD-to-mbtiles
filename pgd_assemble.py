import gc
import io
import json
import os
import struct
import math
import html
import numpy as np
import cv2
import click
from typing import Tuple, Generator, Optional, Sequence
from tkinter import Tk
from tkinter import filedialog
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Отключить предупреждение DecompressionBombWarning
from pgd_geowarp_mbtiles import export_geowarp_mbtiles, build_lower_zoom_pyramid

"""
Константы формата PGD (по результатам реверс-инжиниринга Java-кода).
MAGIC_FILE — сигнатура файла, MAGIC_NODE — сигнатура узла,
MAGIC_ENTRIES_LIST/TABLE — сигнатуры списков и таблиц, MAGIC_DATA_MAIN/ADD — сигнатуры блоков данных.
"""
MAGIC_FILE = 0x5047443A
MAGIC_NODE = 87381
MAGIC_ENTRIES_LIST = 152917
MAGIC_ENTRIES_TABLE = 283989
MAGIC_DATA_MAIN = 1070421
MAGIC_DATA_ADD = 2118997

def wgs84_to_mercator(lat, lon):
    R = 6378137.0
    x = lon * R * math.pi / 180.0
    y = math.log(math.tan((90 + lat) * math.pi / 360.0)) * R
    return x, y

def be_u32(b):
    return struct.unpack('>I', b)[0]


def be_u64(b):
    return struct.unpack('>Q', b)[0]


class FileView:
    """
    Класс-обёртка для работы с бинарным файлом PGD: чтение по смещению, чтение int/long.
    """
    def __init__(self, path):
        self.path = path
        self.f = open(path, 'rb')

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

    def read_at(self, pos, size):
        self.f.seek(pos)
        return self.f.read(size)

    def u32_at(self, pos):
        return be_u32(self.read_at(pos, 4))

    def u64_at(self, pos):
        return be_u64(self.read_at(pos, 8))
        import gc



class NodeListPage:
    """
    Страница списка узлов (используется для хранения дочерних элементов и данных).
    """
    def __init__(self, base, capacity, child_count, data_count, next_ptr):
        self.base = base
        self.capacity = capacity
        self.child_count = child_count
        self.data_count = data_count
        self.next_ptr = next_ptr
        self.entries_pos = base + 24


class NodeList:
    """
    Список страниц с дочерними узлами и данными (цепочка NodeListPage).
    """
    def __init__(self, pages):
        self.pages = pages

    def total_children(self):
        return sum(p.child_count for p in self.pages)

    def total_data(self):
        return sum(p.data_count for p in self.pages)

    def child_slot_pos(self, idx):
        # idx is 0-based across the chain; find page and slot
        off = idx
        for p in self.pages:
            if off < p.child_count:
                return p.entries_pos + off * 12
            off -= p.child_count
        return None

    def data_slot_pos(self, idx):
        # idx is 0-based across the chain; data are stored reversed at page tail
        off = idx
        for p in self.pages:
            if off < p.data_count:
                total_slots = p.capacity
                # data indices are placed from the end, contiguous
                slot_index = (total_slots - 1) - off
                return p.entries_pos + slot_index * 12
            off -= p.data_count
        return None


class NodeTable:
    """
    Таблица дочерних узлов и данных (используется для хранения тайлов).
    """
    def __init__(self, base, child_count, data_count):
        self.base = base
        self.child_count = child_count
        self.data_count = data_count
        self.entries_pos = base + 12

    def child_slot_pos(self, idx):
        if 0 <= idx < self.child_count:
            return self.entries_pos + idx * 12
        return None

    def data_slot_pos(self, idx):
        # tiles are stored reversed at the end: slot_index = total_slots - 1 - idx
        if 0 <= idx < self.data_count:
            total_slots = self.child_count + self.data_count
            slot_index = (total_slots - 1) - idx
            return self.entries_pos + slot_index * 12
        return None


class Node:
    """
    Узел PGD-дерева: содержит указатель, флаги и структуру (NodeList или NodeTable).
    """
    def __init__(self, ptr, flags, entries):
        self.ptr = ptr
        self.flags = flags
        self.entries = entries  # NodeList or NodeTable


# -------------------- Декодирование метаданных (aux) --------------------

class _BytesReader:
    """
    Вспомогательный класс для чтения сериализованных метаданных aux (строки, числа, bool, blob).
    """
    def __init__(self, data: bytes):
        self._b = data
        self._i = 0

    def read(self, n) -> bytes:
        if self._i + n > len(self._b):
            raise EOFError('Unexpected EOF in aux reader')
        out = self._b[self._i:self._i+n]
        self._i += n
        return out

    def read_int(self) -> int:
        return struct.unpack('>i', self.read(4))[0]

    def read_long(self) -> int:
        return struct.unpack('>q', self.read(8))[0]

    def read_double(self) -> float:
        return struct.unpack('>d', self.read(8))[0]

    def read_bool(self) -> bool:
        b = self.read(1)[0]
        return b != 0

    def read_string(self) -> str | None:
        ln = self.read_int()
        if ln == 0:
            return None
        s = self.read(ln).decode('utf-8')
        return s


def read_node_metadata(fv: FileView, node: Node) -> dict:
    """
    Читает метаданные (aux) из узла PGD, возвращает словарь ключ-значение.
    """
    meta_ptr = fv.u64_at(node.ptr + 8)
    if meta_ptr == 0:
        return {}
    body = read_data_chain(fv, meta_ptr)
    if not body:
        return {}
    r = _BytesReader(body)
    out = {}
    try:
        count = r.read_int()
        for _ in range(count):
            key = r.read_string()
            t = r.read_int()
            if t >= 0:
                # string (UTF-8, length t)
                val = r.read(t).decode('utf-8') if t > 0 else ''
            elif t == -1:
                val = r.read_bool()
            elif t == -2:
                val = r.read_long()
            elif t == -3:
                val = r.read_double()
            elif t == -4:
                # blob: int length + bytes
                blen = r.read_int()
                val = r.read(blen) if blen > 0 else b''
            else:
                val = None
            if key is not None and val is not None:
                out[key] = val
    except Exception:
        # tolerate metadata parsing issues
        pass
    return out


def parse_node(fv: FileView, ptr):
    """
    Парсит узел PGD по смещению: определяет тип (NodeList или NodeTable), возвращает Node.
    """
    if ptr <= 0:
        return None
    magic = fv.u32_at(ptr)
    if magic != MAGIC_NODE:
        raise IOError(f'Bad node signature at #{ptr}: {magic}')
    flags = fv.u32_at(ptr + 4)
    # meta_ptr = fv.u64_at(ptr + 8)  # not needed here
    entries_ptr = fv.u64_at(ptr + 16)
    sig = fv.u32_at(entries_ptr)
    if sig == MAGIC_ENTRIES_TABLE:
        child_count = fv.u32_at(entries_ptr + 4)
        data_count = fv.u32_at(entries_ptr + 8)
        return Node(ptr, flags, NodeTable(entries_ptr, child_count, data_count))
    elif sig == MAGIC_ENTRIES_LIST:
        pages = []
        page_ptr = entries_ptr
        while page_ptr:
            sig2 = fv.u32_at(page_ptr)
            if sig2 != MAGIC_ENTRIES_LIST:
                raise IOError(f'Bad list page signature at #{page_ptr}: {sig2}')
            capacity = fv.u32_at(page_ptr + 4)
            child_count = fv.u32_at(page_ptr + 8)
            data_count = fv.u32_at(page_ptr + 12)
            next_ptr = fv.u64_at(page_ptr + 16)
            pages.append(NodeListPage(page_ptr, capacity, child_count, data_count, next_ptr))
            page_ptr = next_ptr
        return Node(ptr, flags, NodeList(pages))
    else:
        raise IOError(f'Unrecognized entries signature #{sig} at {entries_ptr}')


def read_slot_ptr_uid(fv: FileView, slot_pos):
    """
    Читает указатель и uid из слота (используется для перехода по дереву PGD).
    """
    if slot_pos is None:
        return 0, 0
    p = fv.u64_at(slot_pos)
    uid = fv.u32_at(slot_pos + 8)
    return p, uid


def read_data_chain(fv: FileView, data_ptr):
    """
    Читает цепочку связанных блоков данных, возвращает полный payload (байты).
    Основной блок: [u32 1070421][u32 id][u64 total_size][u64 seg_size][u64 next_ptr] ...
    Дополнительные блоки: [u32 2118997][u64 seg_size][u64 next_ptr] ...
    """
    if data_ptr <= 0:
        return b''
    pos = data_ptr
    sig = fv.u32_at(pos)
    if sig != MAGIC_DATA_MAIN:
        raise IOError(f'Bad data signature at #{pos}: {sig}')
    # id_ = fv.u32_at(pos + 4)
    total = fv.u64_at(pos + 8)
    seg = fv.u64_at(pos + 16)
    next_ptr = fv.u64_at(pos + 24)
    payload = bytearray()
    cur = pos + 32
    take = min(seg, total)
    payload += fv.read_at(cur, take)
    remain = total - take
    while remain > 0:
        if next_ptr <= 0:
            raise IOError('Corrupted archive: data chain cut')
        pos = next_ptr
        sig = fv.u32_at(pos)
        if sig != MAGIC_DATA_ADD:
            raise IOError(f'Bad data addition signature at #{pos}: {sig}')
        seg = fv.u64_at(pos + 4)
        next_ptr = fv.u64_at(pos + 12)
        cur = pos + 20
        take = min(seg, remain)
        payload += fv.read_at(cur, take)
        remain -= take
    return bytes(payload)

def extract_tile_image_bytes(fv: FileView, slot_pos):
    """
    Извлекает байты изображения тайла из слота PGD (с учётом кастомного заголовка).
    """
    data_ptr, _uid = read_slot_ptr_uid(fv, slot_pos)
    if data_ptr <= 0:
        return None
    raw = read_data_chain(fv, data_ptr)
    if len(raw) < 2:
        return None
    # Strip custom 2+N header: read = (b0<<1) + b1
    skip = ((raw[0] & 0xFF) << 1) + (raw[1] & 0xFF)
    body = raw[2 + skip:]
    return body

def digits_to_int_lsd_first(digits):
    """
    Преобразует список цифр (младший разряд первым) в целое число.
    """
    val = 0
    mul = 1
    for d in digits:
        val += d * mul
        mul *= 10
    return val

def get_child_node_from_list(fv: FileView, nlist: NodeList, idx):
    """
    Получает дочерний узел по индексу из NodeList.
    """
    slot = nlist.child_slot_pos(idx)
    if slot is None:
        return None
    child_ptr, _ = read_slot_ptr_uid(fv, slot)
    if child_ptr <= 0:
        return None
    return parse_node(fv, child_ptr)


def get_child_node_from_table(fv: FileView, ntable: NodeTable, idx):
    """
    Получает дочерний узел по индексу из NodeTable.
    """
    slot = ntable.child_slot_pos(idx)
    if slot is None:
        return None
    child_ptr, _ = read_slot_ptr_uid(fv, slot)
    if child_ptr <= 0:
        return None
    return parse_node(fv, child_ptr)


def get_tile_from_table(fv: FileView, ntable: NodeTable, idx):
    """
    Извлекает байты изображения тайла по индексу из NodeTable.
    """
    slot = ntable.data_slot_pos(idx)
    if slot is None:
        return None
    return extract_tile_image_bytes(fv, slot)

def enumerate_tiles(fv: FileView, level_node: Node):
    """
    Рекурсивно извлекает все тайлы из узла уровня PGD.
    Возвращает список (x, y, bytes).
    Квадранты: 0:(+,+), 1:(+,-), 2:(-,+), 3:(-,-).
    Вложенные walk_x/walk_y обходят дерево по X и Y.
    """
    # level_node is a TL (list of 4 quadrant TN roots at indices 0..3)
    if not isinstance(level_node.entries, NodeList):
        raise ValueError('Level node must be a list of quadrants')

    tiles = []  # list of (x, y, bytes)

    # Quadrant mapping: 0:(+,+), 1:(+,-), 2:(-,+), 3:(-,-)
    quadrants = [
        (0, 1, 1),
        (1, 1, -1),
        (2, -1, 1),
        (3, -1, -1),
    ]

    for idx, x_sign, y_sign in quadrants:
        qnode = get_child_node_from_list(fv, level_node.entries, idx)
        if qnode is None or not isinstance(qnode.entries, NodeTable):
            continue

        # Traverse X digits first
        def walk_x(node_table: NodeTable, x_digits):
            # deeper x digits (children 0..9)
            for d in range(10):
                child = get_child_node_from_table(fv, node_table, d)
                if child and isinstance(child.entries, NodeTable):
                    walk_x(child.entries, x_digits + [d])

            # last X digit to reach Y-branch (children 10..19)
            for d in range(10):
                y_node = get_child_node_from_table(fv, node_table, 10 + d)
                if y_node and isinstance(y_node.entries, NodeTable):
                    walk_y(y_node.entries, x_digits + [d], [])

        def walk_y(node_table: NodeTable, x_digits, y_digits):
            # deeper Y digits (children 0..9)
            for d in range(10):
                child = get_child_node_from_table(fv, node_table, d)
                if child and isinstance(child.entries, NodeTable):
                    walk_y(child.entries, x_digits, y_digits + [d])

            # leaf tiles (indices 0..9)
            for d in range(10):
                data = get_tile_from_table(fv, node_table, d)
                if data:
                    x = x_sign * digits_to_int_lsd_first(x_digits)
                    y = y_sign * digits_to_int_lsd_first(y_digits + [d])
                    tiles.append((x, y, data))

        walk_x(qnode.entries, [])

    return tiles

def enumerate_tile_slots(fv: FileView, level_node: Node) -> Generator[Tuple[int, int, int], None, None]:
    """
    Ленивая генерация позиций слотов тайлов без чтения самих данных.
    Для каждого тайла отдаёт кортеж (x, y, slot_pos), где slot_pos — абсолютная позиция в файле.
    """
    if not isinstance(level_node.entries, NodeList):
        raise ValueError('Level node must be a list of quadrants')

    # Quadrant mapping consistent with enumerate_tiles
    quadrants = [
        (0, 1, 1),
        (1, 1, -1),
        (2, -1, 1),
        (3, -1, -1),
    ]

    for idx, x_sign, y_sign in quadrants:
        qnode = get_child_node_from_list(fv, level_node.entries, idx)
        if qnode is None or not isinstance(qnode.entries, NodeTable):
            continue

        def walk_x(node_table: NodeTable, x_digits):
            # deeper x digits (children 0..9)
            for d in range(10):
                child = get_child_node_from_table(fv, node_table, d)
                if child and isinstance(child.entries, NodeTable):
                    yield from walk_x(child.entries, x_digits + [d])

            # last X digit to reach Y-branch (children 10..19)
            for d in range(10):
                y_node = get_child_node_from_table(fv, node_table, 10 + d)
                if y_node and isinstance(y_node.entries, NodeTable):
                    yield from walk_y(y_node.entries, x_digits + [d], [])

        def walk_y(node_table: NodeTable, x_digits, y_digits):
            # deeper Y digits (children 0..9)
            for d in range(10):
                child = get_child_node_from_table(fv, node_table, d)
                if child and isinstance(child.entries, NodeTable):
                    yield from walk_y(child.entries, x_digits, y_digits + [d])

            # leaf tiles (indices 0..9) — collect slot positions if present
            for d in range(10):
                slot = node_table.data_slot_pos(d)
                if slot is None:
                    continue
                data_ptr, _ = read_slot_ptr_uid(fv, slot)
                if data_ptr and data_ptr > 0:
                    x = x_sign * digits_to_int_lsd_first(x_digits)
                    y = y_sign * digits_to_int_lsd_first(y_digits + [d])
                    yield (x, y, slot)
        yield from walk_x(qnode.entries, [])

def list_level_nodes(fv: FileView, root_node: Node):
    """
    Возвращает список узлов уровней (LOD) под первым каналом.
    Многие PGD содержат несколько уровней детализации (LOD) в виде списка.
    Обычно последний — максимальная детализация.
    """
    if not isinstance(root_node.entries, NodeList):
        return []
    channel = get_child_node_from_list(fv, root_node.entries, 0)
    if channel is None or not isinstance(channel.entries, NodeList):
        return []
    # Iterate through all children of the channel list
    levels = []
    child_total = channel.entries.total_children()
    for idx in range(child_total):
        lvl = get_child_node_from_list(fv, channel.entries, idx)
        if lvl is not None:
            levels.append(lvl)
    return levels

def find_first_level_node(fv: FileView, root_node: Node):
    """
    Оставлено для обратной совместимости; теперь возвращает первый доступный уровень
    """
    levels = list_level_nodes(fv, root_node)
    return levels[0] if levels else None

def find_highest_detail_level_node(fv: FileView, root_node: Node):
    """
    Эвристика: вернуть последний уровень в списке, обычно это максимальная детализация.
    """
    levels = list_level_nodes(fv, root_node)
    return levels[-1] if levels else None

def assemble_image(fv: FileView, level_node: Node, tile_size: int):
    """
    Собирает мозаику, читая тайлы по одному, чтобы уменьшить пиковое использование памяти.
    Возвращает (canvas, tile_count).
    """
    if Image is None:
        raise RuntimeError('Pillow not available. Please install pillow to decode tiles.')

    min_x = None
    min_y = None
    max_x = None
    max_y = None
    tile_count = 0

    for x, y, _slot in enumerate_tile_slots(fv, level_node):
        tile_count += 1
        min_x = x if min_x is None else min(min_x, x)
        min_y = y if min_y is None else min(min_y, y)
        max_x = x if max_x is None else max(max_x, x)
        max_y = y if max_y is None else max(max_y, y)

    if tile_count == 0 or min_x is None or min_y is None or max_x is None or max_y is None:
        raise RuntimeError('No tiles found')

    tile_w = tile_h = tile_size
    cols = max_x - min_x + 1
    rows = max_y - min_y + 1
    out_w = cols * tile_w
    out_h = rows * tile_h

    canvas = Image.new('RGBA', (out_w, out_h), (0, 0, 0, 0))
    spinner = ['|', '/', '-', '\\']

    for idx, (x, y, slot) in enumerate(enumerate_tile_slots(fv, level_node)):
        data = extract_tile_image_bytes(fv, slot)
        if not data:
            continue
        try:
            with Image.open(io.BytesIO(data)) as img:
                tile_img = img.convert('RGBA')
        except Exception:
            raise RuntimeError('Ошибка разбора тайла')
        cx = (x - min_x) * tile_w
        cy = (y - min_y) * tile_h
        canvas.paste(tile_img, (cx, cy))
        spin = spinner[idx % len(spinner)]
        print(f"\rОбработка тайлов ({idx+1}/{tile_count}) {spin}", end='')
    print()
    return canvas, tile_count

def load_root_node(fv: FileView):
    # Заголовок файла: [u32 magic][u32 version][u64 root_ptr]
    magic = fv.u32_at(0)
    if magic != MAGIC_FILE:
        raise IOError(f'Not a PGD file (magic={hex(magic)})')
    version = fv.u32_at(4)
    if version > 1:
        raise IOError(f'Unsupported PGD version {version}')
    root_ptr = fv.u64_at(8)
    return parse_node(fv, root_ptr)

def get_affin_coeffs(fv):
    with open(fv.path, 'rb') as f:
        data = f.read()
    shftprop_sig = b'SHFTPROP'
    idx = data.find(shftprop_sig)
    if idx == -1:
        raise ValueError('SHFTPROP signature not found!')
    offset = idx + len(shftprop_sig)
    def read_str(data: bytes, offset: int) -> tuple[str, int]:
        strlen = struct.unpack('>I', data[offset:offset+4])[0]
        s = data[offset+4:offset+4+strlen].decode('utf-8')
        return s, offset+4+strlen
    epsg, offset = read_str(data, offset)
    extra_info, offset = read_str(data, offset)
    doubles = [struct.unpack('>d', data[offset+i*8:offset+(i+1)*8])[0] for i in range(18)]
    return doubles

def parse_pgdco(meta):
    co = meta.get('PGD_CO') if isinstance(meta, dict) else None
    if isinstance(co, str) and co.strip():
        pts = []
        for token in co.strip().split():
            if ',' in token:
                lon_s, lat_s = token.split(',', 1)
                try:
                    lon = float(lon_s)
                    lat = float(lat_s)
                    pts.append((lon, lat))
                except ValueError:
                    pass
        return pts

def mercator_to_lonlat(x: float, y: float) -> Tuple[float, float]:
    """Преобразует координаты Меркатора в долготу и широту."""
    R = 6378137.0
    lon = x / R * 180.0 / math.pi
    lat = (2 * math.atan(math.exp(y / R)) - math.pi / 2) * 180.0 / math.pi
    return lon, lat

def get_canvas_size_bounds(canvas):
    canvas_width, canvas_height = canvas.size
    bounds = [
        (0, 0),  # top left
        (canvas_width, 0),  # top right
        (canvas_width, canvas_height),  # bottom right
        (0, canvas_height)  # bottom left
    ]
    return canvas_width, canvas_height, bounds

def apply_affine_to_points(src_points, affine_coeffs):
    """
    src_points: список из 4 точек [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
    affine_coeffs: список из 9 коэффициентов [a0, a1, a2, a3, a4, a5, a6, a7, a8]
    Возвращает список dst_points после projective (affine) преобразования.
    """
    a = affine_coeffs
    dst_points = []
    for point in src_points:
        x, y = point
        xp = a[0]*x + a[1]*y + a[2]
        yp = a[3]*x + a[4]*y + a[5]
        wp = a[6]*x + a[7]*y + a[8]
        if wp != 0:
            dst_x = xp / wp
            dst_y = yp / wp
        else:
            dst_x = float('nan')
            dst_y = float('nan')
        dst_points.append((dst_x, dst_y))
    return dst_points

def save_bounds(output_base, bounds):
    bounds_path = f"{output_base}_bounds.json"
    bounds_dir = os.path.dirname(bounds_path)
    if bounds_dir and not os.path.isdir(bounds_dir):
        os.makedirs(bounds_dir, exist_ok=True)
    with open(bounds_path, 'w', encoding='utf-8') as bounds_file:
        json.dump(bounds, bounds_file, ensure_ascii=False, indent=2)
    print(f'[--save-bounds] Границы сохранены в {bounds_path}')

def save_ms(output_base, metadata_name, min_zoom, max_zoom):
    ms_path = f"{output_base}.ms"
    ms_dir = os.path.dirname(ms_path)
    if ms_dir and not os.path.isdir(ms_dir):
        os.makedirs(ms_dir, exist_ok=True)
    ms_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<customMapSource overlay="true" overzoom="true">\n'
        f'    <name>{html.escape(metadata_name)}</name>\n'
        f'    <minZoom>{min_zoom}</minZoom>\n'
        f'    <maxZoom>{max_zoom}</maxZoom>\n'
        '</customMapSource>\n'
    )
    with open(ms_path, 'w', encoding='utf-8') as ms_file:
        ms_file.write(ms_xml)
    print(f'[--save-ms] MapSource сохранён в {ms_path}')


def choose_pgd_files() -> Sequence[str]:
    """Открывает диалог выбора PGD-файлов, возвращает кортеж путей."""
    root = Tk()
    root.withdraw()
    try:
        paths = filedialog.askopenfilenames(
            title='Выберите PGD-файл(ы)',
            filetypes=[('PGD files', '*.pgd'), ('All files', '*.*')],
        )
    finally:
        root.destroy()
    return paths

MBTILES_SCHEMA = '''
CREATE TABLE IF NOT EXISTS tiles (zoom_level INTEGER, tile_column INTEGER, tile_row INTEGER, tile_data BLOB);
CREATE TABLE IF NOT EXISTS metadata (name TEXT, value TEXT);
CREATE UNIQUE INDEX IF NOT EXISTS tile_index on tiles (zoom_level, tile_column, tile_row);
'''

def run_conversion(
    pgd_path: str,
    out_name: Optional[str],
    image_format: str,
    save_mosaic: bool,
    save_bounds_flag: bool,
    save_ms_flag: bool,
    zoom_pad: int,
    no_dedup: bool,
) -> None:
    base = os.path.dirname(__file__)
    image_format = image_format.lower()
    if image_format not in ('webp', 'png'):
        raise ValueError('Поддерживаются только форматы webp или png')
    raw_out_name = out_name or os.path.splitext(os.path.basename(pgd_path))[0]
    out_stem = os.path.splitext(raw_out_name)[0]
    if os.path.isabs(out_stem):
        output_base = os.path.normpath(out_stem)
    else:
        output_base = os.path.normpath(os.path.join(base, out_stem))
    metadata_name = os.path.basename(out_stem) or os.path.splitext(os.path.basename(pgd_path))[0]
    fv = FileView(pgd_path)

    try:
        root = load_root_node(fv)
        if root is None:
            raise RuntimeError('Failed to load root node from PGD')
        level = find_highest_detail_level_node(fv, root)
        if level is None:
            raise RuntimeError('Failed to locate any level node')

        # --- Необходимые метаданные PGD для работы ---
        meta = read_node_metadata(fv, level)
        affin_coeffs = get_affin_coeffs(fv)
        TILE_SIZE = meta['PPTX']

        # --- Собираем мозаику ---
        mosaic, tile_count = assemble_image(fv, level, TILE_SIZE)
        print(f'Размер мозайки: {mosaic.size[0]}x{mosaic.size[1]}')

        # --- Сохраняем мозаику на диск при соответствующем аргументе ---
        if save_mosaic:
            output_format = image_format.upper()
            mosaic_out_path = f"{output_base}.{image_format}"
            mosaic_out_dir = os.path.dirname(mosaic_out_path)
            if mosaic_out_dir and not os.path.isdir(mosaic_out_dir):
                os.makedirs(mosaic_out_dir, exist_ok=True)
            mosaic.save(mosaic_out_path, format=output_format)
            print(f'[--save-mosaic] Мозаика сохранена в {mosaic_out_path}')

        # --- Применяем афинное преобразование, вычисляем матрицу ---
        _, _, srcPoints = get_canvas_size_bounds(mosaic)
        dstPoints = apply_affine_to_points(srcPoints, affin_coeffs[:9])
        src_np = np.array(srcPoints, dtype=np.float32)
        dst_np = np.array(dstPoints, dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_np, dst_np)
        width = int(np.ceil(np.max(dst_np[:, 0]) - np.min(dst_np[:, 0])))
        height = int(np.ceil(np.max(dst_np[:, 1]) - np.min(dst_np[:, 1])))
        offset_x = np.min(dst_np[:, 0])
        offset_y = np.min(dst_np[:, 1])
        # --- Матрица сдвига холста в 0,0 после преобразования ---
        T = np.array([[1, 0, -offset_x], [0, 1, -offset_y], [0, 0, 1]], dtype=np.float32)
        M_shifted = (T @ M).astype(np.float64)

        original_width = mosaic.width
        img_np = np.asarray(mosaic)
        if not img_np.flags['C_CONTIGUOUS']:
            img_np = np.ascontiguousarray(img_np)
        mosaic.close()
        del mosaic
        gc.collect()

        # --- Так как матрица дает px->mercator скейлим холст к исходной ширине ---
        if width != original_width and original_width > 0:
            scale_x = original_width / width
            scale_matrix = np.array([
                [scale_x, 0.0, 0.0],
                [0.0, scale_x, 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            M_shifted = scale_matrix @ M_shifted
            width = original_width
            height = max(1, int(round(height * scale_x)))

        dest_width = int(width)
        dest_height = int(height)
        dest_size = (dest_width, dest_height)
        width = dest_width
        height = dest_height
        M_inv = np.linalg.inv(M_shifted)

        corners = [mercator_to_lonlat(x, y) for x, y in dstPoints]
        corners = [(lat, -lon) for lat, lon in corners]
        lons = [pt[0] for pt in corners]
        lats = [pt[1] for pt in corners]
        bounds = (min(lons), min(lats), max(lons), max(lats))
        bounds_str = "%.8f,%.8f,%.8f,%.8f" % bounds

        lon_coverage = abs(max(lons) - min(lons))
        zoom = -1
        if lon_coverage > 0:
            px_per_deg = width / lon_coverage
            for z in range(34):
                std_px_per_deg = (2 ** z) * 256.0 / 360.0
                if std_px_per_deg > px_per_deg:
                    zoom = z
                    break
            if zoom == -1:
                zoom = int(round(math.log2(max(width, height) / TILE_SIZE)))
        else:
            zoom = int(round(math.log2(max(width, height) / TILE_SIZE)))

        if zoom < 0:
            raise RuntimeError('Не удалось определить уровень зума для MBTiles')

        zoom_pad = max(0, zoom_pad)
        max_zoom = max(0, zoom)
        min_zoom = max(0, max_zoom - zoom_pad)

        center_lat = (min(lats) + max(lats)) / 2.0
        center_lon = (min(lons) + max(lons)) / 2.0
        center_zoom = (min_zoom + max_zoom) // 2
        center = "%.6f,%.6f,%d" % (center_lon, center_lat, center_zoom)

        if save_bounds_flag:
            save_bounds(output_base, bounds)

        if save_ms_flag:
            save_ms(output_base, metadata_name, min_zoom, max_zoom)

        mbtiles_tile_size = 256
        mbtiles_metadata = {
            'name': metadata_name,
            'format': image_format,
            'type': 'overlay',
            'version': '1.0',
            'description': '@soreixe',
            'minzoom': str(min_zoom),
            'maxzoom': str(max_zoom),
            'bounds': bounds_str,
            'center': center,
        }

        mbtiles_path = f"{output_base}.mbtiles"
        mbtiles_dir = os.path.dirname(mbtiles_path)
        if mbtiles_dir and not os.path.isdir(mbtiles_dir):
            os.makedirs(mbtiles_dir, exist_ok=True)
        print(f"Экспорт MBTiles в {mbtiles_path} ...")

        render_min_zoom = max_zoom if min_zoom < max_zoom else min_zoom
        export_geowarp_mbtiles(
            None,
            bounds,
            zoom,
            mbtiles_path,
            tile_size=mbtiles_tile_size,
            metadata=mbtiles_metadata,
            deduplicate=not no_dedup,
            image_format=image_format,
            min_zoom=render_min_zoom,
            max_zoom=max_zoom,
            projective_source={
                'src_array': img_np,
                'matrix_inv': M_inv,
                'dest_size': dest_size,
                'border_value': (0, 0, 0, 0),
            },
        )

        build_lower_zoom_pyramid(
            mbtiles_path,
            min_zoom=min_zoom,
            max_zoom=max_zoom,
            image_format=image_format,
            tile_size=mbtiles_tile_size,
            deduplicate=not no_dedup,
        )

        del img_np
        gc.collect()

    finally:
        fv.close()


@click.command()
@click.option(
    '--pgd',
    'pgd_path',
    default=None,
    type=str,
    help='Путь к входному PGD-файлу. Если не указан, откроется диалог выбора.',
)
@click.option(
    '--out-name',
    default=None,
    type=str,
    help='Базовое имя выходных файлов (без расширения). По умолчанию используется имя PGD-файла.',
)
@click.option(
    '--format',
    'image_format',
    default='webp',
    type=click.Choice(['webp', 'png'], case_sensitive=False),
    show_default=True,
    help='Формат итоговых изображений (webp или png).',
)
@click.option('--save-mosaic', is_flag=True, help='Сохранять промежуточную мозаику на диск.')
@click.option('--save-bounds', 'save_bounds_flag', is_flag=True, help='Сохранить bounds в JSON (например, для Leaflet).')
@click.option('--save-ms', 'save_ms_flag', is_flag=True, help='Создать MapSource (*.ms) для Guru Maps.')
@click.option(
    '--zoom-pad',
    default=0,
    show_default=True,
    type=int,
    help='Количество уровней вниз от автоматически вычисленного зума.',
)
@click.option('--no-dedup', is_flag=True, help='Отключить дедупликацию одинаковых тайлов.')

def main(
    pgd_path: Optional[str],
    out_name: Optional[str],
    image_format: str,
    save_mosaic: bool,
    save_bounds_flag: bool,
    save_ms_flag: bool,
    zoom_pad: int,
    no_dedup: bool,
) -> None:
    """CLI-обёртка для конвертации PGD в MBTiles."""

    targets: Sequence[str]
    if pgd_path:
        targets = (pgd_path,)
    else:
        selected = choose_pgd_files()
        if not selected:
            print('PGD-файлы не выбраны. Выход.')
            return
        targets = selected

    for idx, path in enumerate(targets, start=1):
        if len(targets) > 1:
            print(f"[{idx}/{len(targets)}] {path}")
        run_conversion(
            pgd_path=path,
            out_name=out_name,
            image_format=image_format,
            save_mosaic=save_mosaic,
            save_bounds_flag=save_bounds_flag,
            save_ms_flag=save_ms_flag,
            zoom_pad=zoom_pad,
            no_dedup=no_dedup,
        )


if __name__ == '__main__':
    main()
