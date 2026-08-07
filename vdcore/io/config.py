"""Vehicle configuration save/load with schema versioning.

Saves and loads Vehicle models as JSON with a schema_version field.
Rejects unknown top-level keys on load to catch hand-edited files
that include derived values (e.g. someone adding "wheelbase_mm").
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vdcore.models.hardpoint import Vehicle

CURRENT_SCHEMA_VERSION = 2

_ALLOWED_TOP_KEYS = frozenset(Vehicle.model_fields.keys())


def save_vehicle(vehicle: Vehicle, path: Path) -> None:
    """Serialize a Vehicle to JSON.

    Derived values (track, wheelbase, contact_patch) are not stored —
    they are plain @property on the models or free functions in
    vdcore.geometry.derived.
    """
    data = vehicle.model_dump(mode="python")
    text = json.dumps(data, indent=2, ensure_ascii=False)
    path.write_text(text, encoding="utf-8")


def load_vehicle(path: Path) -> Vehicle:
    """Deserialize a Vehicle from JSON.

    Raises ValueError if the schema version is unsupported or if
    unknown top-level keys are present (likely hand-edited derived values).
    """
    text = path.read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(text)

    version = data.get("schema_version")
    if version is not None and version > CURRENT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema version {version} "
            f"(max supported: {CURRENT_SCHEMA_VERSION}). "
            f"Update vdcore to read this file."
        )

    unknown_keys = set(data.keys()) - _ALLOWED_TOP_KEYS
    if unknown_keys:
        raise ValueError(
            f"Unknown top-level keys in vehicle config: {sorted(unknown_keys)}. "
            f"Derived values like 'wheelbase_mm' and 'track_mm' should not be "
            f"in the config file — they are computed from hardpoints."
        )

    return Vehicle.model_validate(data)
