"""MoTeC i2 ``.ld`` log reader. NumPy only -- no ``ldparser`` dependency.

File layout (little endian), as documented by the open ``ldparser`` project
and checked against the team's sample logs:

* header: ``u32`` marker at 0x00, channel-meta pointer at 0x08, data pointer
  at 0x0C, date at 0x5E (16 s), time at 0x7E (16 s), driver at 0x9E (64 s),
  vehicle at 0xDE (64 s), venue at 0x15E (64 s);
* channel metadata is a doubly linked list. Each block holds
  ``prev, next, data_ptr, n`` (``u32``), ``counter, dtype_a, dtype_size,
  freq`` (``u16``), ``shift, mul, scale, dec`` (``i16``), name (32 s), short
  name (8 s), unit (12 s);
* ``dtype_a`` 0x07 is float (size 2 -> f16, 4 -> f32, 8 -> f64); 0x00, 0x03,
  0x05 are integers (2 -> i16, 4 -> i32);
* engineering value = (raw / scale · 10^-dec + shift) · mul.

The concept notebook in ``random-things`` read the unit from the short-name
slot and applied a different integer scaling; both are corrected here.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_FLOAT = {2: "<f2", 4: "<f4", 8: "<f8"}
_INT = {2: "<i2", 4: "<i4"}


@dataclass
class LdChannel:
    name: str
    unit: str
    freq_hz: float
    data: np.ndarray
    short_name: str = ""


@dataclass
class LdFile:
    path: str
    header: dict[str, str] = field(default_factory=dict)
    channels: list[LdChannel] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)   # (channel, why it was not read)


class LdFormatError(ValueError):
    pass


def _cstr(buf: bytes, off: int, n: int) -> str:
    return buf[off:off + n].split(b"\x00")[0].decode("latin1").strip()


def read_ld(path: str | Path) -> LdFile:
    data = Path(path).read_bytes()
    if len(data) < 0x200:
        raise LdFormatError(f"{Path(path).name}: too short to be a MoTeC .ld file")
    meta_ptr = struct.unpack_from("<I", data, 0x08)[0]
    if not 0 < meta_ptr < len(data):
        raise LdFormatError(f"{Path(path).name}: channel metadata pointer out of range")
    header = {
        "date": _cstr(data, 0x5E, 16), "time": _cstr(data, 0x7E, 16),
        "driver": _cstr(data, 0x9E, 64), "vehicle": _cstr(data, 0xDE, 64),
        "venue": _cstr(data, 0x15E, 64),
    }
    out = LdFile(str(path), header)
    seen: set[int] = set()
    ptr = meta_ptr
    while ptr:
        if ptr in seen or ptr + 84 > len(data):
            raise LdFormatError(f"{Path(path).name}: corrupt channel list at 0x{ptr:x}")
        seen.add(ptr)
        _prev, nxt, dptr, n = struct.unpack_from("<IIII", data, ptr)
        _ctr, dtype_a, size, freq = struct.unpack_from("<HHHH", data, ptr + 16)
        shift, mul, scale, dec = struct.unpack_from("<hhhh", data, ptr + 24)
        name = _cstr(data, ptr + 32, 32)
        short = _cstr(data, ptr + 64, 8)
        unit = _cstr(data, ptr + 72, 12)
        if dtype_a == 0x07:
            fmt = _FLOAT.get(size)
        elif dtype_a in (0x00, 0x03, 0x05):
            fmt = _INT.get(size)
        else:
            fmt = None
        if fmt is None or dptr + n * size > len(data):
            # unknown type or truncated block: skip, never guess -- but say so
            out.skipped.append((name, f"dtype 0x{dtype_a:02x} size {size} not supported"
                                if fmt is None else "truncated data block"))
            ptr = nxt
            continue
        raw = np.frombuffer(data, np.dtype(fmt), n, dptr).astype(float)
        val = (raw / (scale or 1) * 10.0 ** (-dec) + shift) * (mul or 1)
        out.channels.append(LdChannel(name, unit, float(freq), val, short))
        ptr = nxt
    if not out.channels:
        why = "; ".join(f"{n}: {w}" for n, w in out.skipped)
        raise LdFormatError(f"{Path(path).name}: no readable channels" + (f" ({why})" if why
                                                                         else ""))
    return out


def write_ld(path: str | Path, channels: list[LdChannel], header: dict[str, str] | None = None
             ) -> None:
    """Minimal ``.ld`` writer (float32 channels). Used for tests and data export.

    The format stores the rate as an integer number of Hz, so a non-integer
    rate is refused rather than truncated.
    """
    header = header or {}
    for ch in channels:
        if ch.freq_hz != int(ch.freq_hz):
            raise ValueError(f"{ch.name}: .ld stores the sample rate as whole Hz; "
                             f"{ch.freq_hz:g} Hz cannot be written")
    head = bytearray(0x6E2)
    meta0 = len(head)
    block = 124
    data_start = meta0 + block * len(channels)
    struct.pack_into("<I", head, 0x00, 0x40)
    struct.pack_into("<I", head, 0x08, meta0)
    struct.pack_into("<I", head, 0x0C, data_start)
    for off, key, n in ((0x5E, "date", 16), (0x7E, "time", 16), (0x9E, "driver", 64),
                        (0xDE, "vehicle", 64), (0x15E, "venue", 64)):
        b = header.get(key, "").encode("latin1")[: n - 1]
        head[off:off + len(b)] = b
    metas = bytearray()
    blobs = bytearray()
    for i, ch in enumerate(channels):
        ptr = meta0 + i * block
        prev = 0 if i == 0 else ptr - block
        nxt = 0 if i == len(channels) - 1 else ptr + block
        arr = np.asarray(ch.data, dtype="<f4")
        dptr = data_start + len(blobs)
        m = bytearray(block)
        struct.pack_into("<IIII", m, 0, prev, nxt, dptr, len(arr))
        struct.pack_into("<HHHH", m, 16, i + 1, 0x07, 4, int(ch.freq_hz))
        struct.pack_into("<hhhh", m, 24, 0, 1, 1, 0)
        for off, s, n in ((32, ch.name, 32), (64, ch.short_name, 8), (72, ch.unit, 12)):
            b = s.encode("latin1")[: n - 1]
            m[off:off + len(b)] = b
        metas += m
        blobs += arr.tobytes()
    Path(path).write_bytes(bytes(head + metas + blobs))
