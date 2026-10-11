"""A loaded telemetry session on one common time base.

Loggers record channels at different rates; every analysis here needs them
aligned. The session resamples every channel onto the fastest channel's time
base by linear interpolation (no extrapolation: samples outside a channel's
own span are NaN). The original rate is kept per channel so the quality
report can still judge each one on its own terms.
"""

from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .ld import read_ld


@dataclass
class Session:
    name: str
    time_s: np.ndarray
    channels: dict[str, np.ndarray] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    rates_hz: dict[str, float] = field(default_factory=dict)
    meta: dict[str, str] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0]) if len(self.time_s) > 1 else 0.0

    @property
    def rate_hz(self) -> float:
        if len(self.time_s) < 2:
            return float("nan")
        return float(1.0 / np.median(np.diff(self.time_s)))

    def names(self) -> list[str]:
        return list(self.channels)

    def get(self, name: str | None) -> np.ndarray | None:
        if not name:
            return None
        return self.channels.get(name)


def _resample(t_src: np.ndarray, y: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    out = np.interp(t_dst, t_src, y)
    out[(t_dst < t_src[0] - 1e-9) | (t_dst > t_src[-1] + 1e-9)] = np.nan
    return out


def load_ld(path: str | Path) -> Session:
    ld = read_ld(path)
    fmax = max(c.freq_hz for c in ld.channels if c.freq_hz > 0)
    n_max = max(len(c.data) / c.freq_hz for c in ld.channels if c.freq_hz > 0)
    t = np.arange(int(round(n_max * fmax))) / fmax
    s = Session(Path(path).stem, t, meta={k: v for k, v in ld.header.items() if v})
    s.meta["source"] = str(path)
    for c in ld.channels:
        if c.freq_hz <= 0 or len(c.data) == 0:
            continue
        tc = np.arange(len(c.data)) / c.freq_hz
        name = c.name
        k = 2
        while name in s.channels:
            name = f"{c.name} ({k})"
            k += 1
        s.channels[name] = c.data.copy() if c.freq_hz == fmax and len(c.data) == len(t) \
            else _resample(tc, c.data, t)
        s.units[name] = c.unit
        s.rates_hz[name] = c.freq_hz
    return s


_TIME_RE = re.compile(r"^(time|t|timestamp|time_s|time \[s\]|time\(s\)|seconds?)$", re.I)


def _sniff(text: str) -> str:
    try:
        return csv.Sniffer().sniff(text[:4096], delimiters=",;\t").delimiter
    except csv.Error:
        return ","


def load_csv(path: str | Path, rate_hz: float | None = None) -> Session:
    """Generic CSV/TSV: one header row (optionally a units row), numeric columns.

    MoTeC i2 CSV exports carry metadata lines above the header; those are
    skipped by looking for the first row that is followed by numeric rows.
    The time column is the one named like 'time'; failing that the first
    strictly increasing column; failing that ``rate_hz`` (required then).
    Decimal commas (pt-BR Excel) are accepted when the delimiter is ';'.
    """
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    delim = _sniff(raw)
    rows = list(csv.reader(io.StringIO(raw), delimiter=delim))

    def num(v: str) -> float | None:
        v = v.strip().strip('"')
        if delim == ";":
            v = v.replace(",", ".")
        try:
            return float(v)
        except ValueError:
            return None

    def numeric_row(r: list[str]) -> bool:
        vals = [num(v) for v in r if v.strip() != ""]
        return len(vals) >= 2 and sum(v is not None for v in vals) >= 0.8 * len(vals)

    header_idx = None
    for i in range(min(len(rows) - 1, 200)):
        if rows[i] and not numeric_row(rows[i]) and any(numeric_row(r) for r in rows[i + 1:i + 3]):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"{Path(path).name}: could not find a header row followed by numbers")
    header = [h.strip().strip('"') or f"col{j}" for j, h in enumerate(rows[header_idx])]
    start = header_idx + 1
    units = [""] * len(header)
    if start < len(rows) and not numeric_row(rows[start]):
        units = [u.strip().strip('"') for u in rows[start]] + [""] * len(header)
        start += 1
    data = [r for r in rows[start:] if r and numeric_row(r)]
    if not data:
        raise ValueError(f"{Path(path).name}: no numeric rows")
    ncol = len(header)
    arr = np.full((len(data), ncol), np.nan)
    for i, r in enumerate(data):
        for j in range(min(ncol, len(r))):
            v = num(r[j])
            if v is not None:
                arr[i, j] = v
    tcol = next((j for j, h in enumerate(header) if _TIME_RE.match(h)), None)
    if tcol is None:
        for j in range(ncol):
            col = arr[:, j]
            if np.all(np.isfinite(col)) and np.all(np.diff(col) > 0):
                tcol = j
                break
    if tcol is not None:
        t = arr[:, tcol] - arr[0, tcol]
    elif rate_hz:
        t = np.arange(len(data)) / rate_hz
    else:
        raise ValueError(f"{Path(path).name}: no time column; give the sample rate")
    s = Session(Path(path).stem, t, meta={"source": str(path)})
    rate = float(1.0 / np.median(np.diff(t))) if len(t) > 1 else math.nan
    for j, h in enumerate(header):
        if j == tcol:
            continue
        col = arr[:, j]
        if not np.isfinite(col).any():
            continue
        s.channels[h] = col
        s.units[h] = units[j] if j < len(units) else ""
        s.rates_hz[h] = rate
    return s


def load_session(path: str | Path, rate_hz: float | None = None) -> Session:
    p = Path(path)
    if p.suffix.lower() == ".ld":
        return load_ld(p)
    return load_csv(p, rate_hz)
