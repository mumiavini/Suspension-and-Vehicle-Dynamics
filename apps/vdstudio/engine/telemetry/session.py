"""A loaded telemetry session on one common time base.

Loggers record channels at different rates; every analysis here needs them
aligned. The session resamples every channel onto the fastest channel's time
base by linear interpolation (no extrapolation: samples outside a channel's
own span are NaN). The original rate is kept per channel, and the original
samples of every resampled channel are kept in ``native``, so the quality
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
    # original samples of channels that were resampled onto ``time_s``
    native: dict[str, np.ndarray] = field(default_factory=dict)
    # channels present in the file but not read: (name, reason)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0]) if len(self.time_s) > 1 else 0.0

    @property
    def rate_hz(self) -> float:
        if len(self.time_s) < 2:
            return float("nan")
        return float(1.0 / np.nanmedian(np.diff(self.time_s)))

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
    s.skipped = list(ld.skipped)
    for c in ld.channels:
        if c.freq_hz <= 0 or len(c.data) == 0:
            s.skipped.append((c.name, "no samples" if c.freq_hz > 0 else "sample rate 0 Hz"))
            continue
        tc = np.arange(len(c.data)) / c.freq_hz
        name = c.name
        k = 2
        while name in s.channels:
            name = f"{c.name} ({k})"
            k += 1
        if c.freq_hz == fmax and len(c.data) == len(t):
            s.channels[name] = c.data.copy()
        else:
            s.channels[name] = _resample(tc, c.data, t)
            s.native[name] = c.data.copy()
        s.units[name] = c.unit
        s.rates_hz[name] = c.freq_hz
    if s.skipped:
        s.meta["skipped channels"] = f"{len(s.skipped)} (see Data quality)"
    return s


_TIME_RE = re.compile(
    r"^(time|t|timestamp|seconds?)(_(s|ms|us)|\s*[\[(]\s*(s|ms|us|µs)\s*[\])])?$", re.I)
# time units -> seconds; the unit comes from the units row, else from the header
_TIME_UNIT = {"": 1.0, "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
              "ms": 1e-3, "msec": 1e-3, "us": 1e-6, "µs": 1e-6, "μs": 1e-6, "usec": 1e-6}
_TIME_UNIT_IN_NAME = re.compile(r"[\s_\[(](s|sec|ms|msec|us|µs|μs)[\])]?$", re.I)
# columns never taken as the time base by the "strictly increasing" fallback
_NOT_TIME_RE = re.compile(r"(dist|lap|odo)", re.I)
# first cells of MoTeC i2 CSV metadata rows (never a header, never data)
_META_KEYS = {"format", "venue", "vehicle", "driver", "device", "comment", "log date",
              "log time", "sample rate", "duration", "range", "beacon markers", "session",
              "short comment", "event"}
_DT_MIN_S, _DT_MAX_S = 1e-5, 0.5       # plausible logger sample interval


def _sniff(text: str) -> str:
    try:
        return csv.Sniffer().sniff(text[:4096], delimiters=",;\t").delimiter
    except csv.Error:
        return ","


def _time_scale(header: str, unit: str) -> tuple[float, str, str]:
    """(factor to seconds, unit label, unit named in the header) for the time column."""
    m = _TIME_UNIT_IN_NAME.search(header.strip())
    in_name = m.group(1).lower() if m else ""
    u = unit.strip().lower() or in_name
    if u in _TIME_UNIT:
        return _TIME_UNIT[u], u or "s (assumed)", in_name
    return 1.0, f"'{unit}' unknown, s assumed", in_name


def load_csv(path: str | Path, rate_hz: float | None = None) -> Session:
    """Generic CSV/TSV: one header row (optionally a units row), numeric columns.

    MoTeC i2 CSV exports carry metadata lines above the header; the header is
    the last non-numeric row before a run of numeric rows as wide as itself
    (blank rows and one units row may sit in between; rows whose first cell is
    a known MoTeC metadata key are never header or data).
    The time column is the one named like 'time'; failing that the first
    strictly increasing column not named like a distance or lap; failing that
    ``rate_hz`` (required then). The time column is scaled by its unit (units
    row, else a ``_ms`` / ``[ms]`` suffix in its name); rows without a time are
    dropped, duplicated time rows are dropped, and a time reset is stitched
    on (each later segment continues one sample after the previous one). All
    of it is recorded in ``meta``. A sample interval outside 10 µs … 0.5 s is
    refused as an implausible time base.
    Decimal commas (pt-BR Excel) are accepted for any delimiter except ','.
    """
    name = Path(path).name
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    delim = _sniff(raw)
    rows = list(csv.reader(io.StringIO(raw), delimiter=delim))

    def num(v: str) -> float | None:
        v = v.strip().strip('"')
        if delim != ",":
            v = v.replace(",", ".")
        try:
            return float(v)
        except ValueError:
            return None

    def numeric_row(r: list[str]) -> bool:
        vals = [num(v) for v in r if v.strip() != ""]
        return len(vals) >= 2 and sum(v is not None for v in vals) >= 0.8 * len(vals)

    def blank(r: list[str]) -> bool:
        return not any(v.strip() for v in r)

    def meta_row(r: list[str]) -> bool:
        return bool(r) and r[0].strip().strip('"').strip().lower() in _META_KEYS

    def next_filled(i: int) -> int | None:
        return next((k for k in range(i, len(rows)) if not blank(rows[k])), None)

    def data_run(i: int, width: int) -> tuple[int, bool]:
        """(numeric rows of ~``width`` fields from row i, blanks skipped; reached EOF)."""
        run = 0
        for r in rows[i:]:
            if blank(r):
                continue
            if meta_row(r) or not numeric_row(r) or abs(len(r) - width) > 1:
                return run, False
            run += 1
            if run >= 10:
                return run, False
        return run, True

    header_idx = units_idx = None
    for i in range(min(len(rows) - 1, 200)):
        r = rows[i]
        if blank(r) or meta_row(r) or numeric_row(r):
            continue
        j = next_filled(i + 1)
        if j is None:
            break
        k: int | None = j
        if not numeric_row(rows[j]):
            if meta_row(rows[j]):
                continue
            k = next_filled(j + 1)          # rows[j] is the units row
            if k is None:
                continue
        run, eof = data_run(k, len(r))
        if run >= 10 or (eof and run > 0):
            header_idx, units_idx = i, (j if k != j else None)
            break
    if header_idx is None:
        raise ValueError(f"{name}: could not find a header row followed by numbers")
    header = [h.strip().strip('"') or f"col{j}" for j, h in enumerate(rows[header_idx])]
    units = [""] * len(header)
    if units_idx is not None:
        units = [u.strip().strip('"') for u in rows[units_idx]] + [""] * len(header)
    start = (units_idx if units_idx is not None else header_idx) + 1
    data = [r for r in rows[start:] if r and not meta_row(r) and numeric_row(r)]
    if not data:
        raise ValueError(f"{name}: no numeric rows")
    ncol = len(header)
    arr = np.full((len(data), ncol), np.nan)
    for i, r in enumerate(data):
        for j in range(min(ncol, len(r))):
            v = num(r[j])
            if v is not None:
                arr[i, j] = v
    meta = {"source": str(path)}
    tcol = next((j for j, h in enumerate(header) if _TIME_RE.match(h)), None)
    if tcol is None:
        for j in range(ncol):
            col = arr[:, j]
            if _NOT_TIME_RE.search(header[j]):
                continue
            if np.all(np.isfinite(col)) and np.all(np.diff(col) > 0):
                tcol = j
                break
    if tcol is not None:
        good = np.isfinite(arr[:, tcol])
        if not good.all():
            arr = arr[good]
            meta["dropped rows (no time)"] = str(int((~good).sum()))
        if len(arr) < 2:
            raise ValueError(f"{name}: fewer than 2 rows with a valid time")
        k, tu, in_name = _time_scale(header[tcol], units[tcol])
        meta["time column"] = f"{header[tcol]} [{tu}]"
        t = arr[:, tcol] * k
        dup = np.concatenate([[False], np.diff(t) == 0.0])
        if dup.any():
            arr, t = arr[~dup], t[~dup]
            meta["dropped rows (duplicate time)"] = str(int(dup.sum()))
        if len(t) < 2:
            raise ValueError(f"{name}: fewer than 2 rows with a distinct time")
        dt = np.diff(t)
        back = np.flatnonzero(dt < 0)
        if back.size:
            step = float(np.median(dt[dt > 0])) if np.any(dt > 0) else math.nan
            if not math.isfinite(step):
                raise ValueError(f"{name}: time column '{header[tcol]}' never increases")
            t = t.copy()
            for i in back:               # later diffs are unchanged, so indices stay valid
                t[i + 1:] += t[i] + step - t[i + 1]
            meta["time resets"] = (f"{back.size} at data row(s) "
                                   f"{', '.join(str(int(i) + 1) for i in back[:5])}"
                                   f"{' …' if back.size > 5 else ''}; later segments appended")
        t = t - t[0]
        dt_med = float(np.median(np.diff(t)))
        if not (math.isfinite(dt_med) and _DT_MIN_S <= dt_med <= _DT_MAX_S):
            clash = (f" (the units row says '{units[tcol]}', the name says {in_name})"
                     if in_name and units[tcol].strip() and in_name != tu else "")
            raise ValueError(f"{name}: implausible time base: dt = {dt_med:g} s in column "
                             f"'{header[tcol]}' read as [{tu}]{clash} -- check the time unit")
    elif rate_hz:
        t = np.arange(len(data)) / rate_hz
        meta["time column"] = f"none (rate {rate_hz:g} Hz given)"
    else:
        raise ValueError(f"{name}: no time column; give the sample rate")
    s = Session(Path(path).stem, t, meta=meta)
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
