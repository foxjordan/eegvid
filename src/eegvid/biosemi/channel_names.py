"""Electrode names from an ActiView configuration (.cfg) file.

The AD-box only knows input slots (A1..A32, B1.., EXG1-8). ActiView maps them to
cap positions with the [Labels] section of its .cfg file:

    Chan1 = Fp1      -> slot A1  (Chan33 -> B1, Chan65 -> C1, ...)
    Tou1 = EXG1      -> slot EXG1 (the 8 touchproof EXG inputs)

The same .cfg files work here, so a lab keeps one source of truth for its caps.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from .protocol import Channel, electrode_label

_KEY = re.compile(r"^(Chan|Tou)(\d+)$", re.IGNORECASE)
BDF_LABEL_MAX = 16  # characters in a BDF channel label


def parse_actiview_cfg(path: str | Path) -> dict[str, str]:
    """Return {slot: name} from the [Labels] section, e.g. {"A1": "Fp1", "EXG1": "EXG1"}."""
    text = Path(path).read_text(encoding="latin-1")
    names: dict[str, str] = {}
    section = None
    for line in text.splitlines():
        line = line.split("//", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            continue
        if section != "labels" or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        value = value.strip("'\"").strip()
        m = _KEY.match(key)
        if not m or not value:
            continue
        n = int(m.group(2))
        slot = electrode_label(n - 1) if m.group(1).lower() == "chan" else f"EXG{n}"
        names[slot] = value[:BDF_LABEL_MAX]
    if not names:
        raise ValueError(f"No channel labels (Chan1 = …) found in {Path(path).name}")
    return names


def apply_names(channels: list[Channel], names: dict[str, str] | None) -> tuple[list[Channel], list[str]]:
    """Rename channels by slot. Returns (channels, notes about duplicate names).

    Channel names must be unique for BDF readers and analysis tools. ActiView .cfg
    files often name the A bank with 10-20 positions while later banks keep their
    slot names, so e.g. A8 = "C3" collides with bank C's third input "C3". A
    later duplicate is renamed to "<name> [<slot>]", trimmed to fit the BDF label.
    """
    if not names:
        return list(channels), []
    out, seen, notes = [], set(), []
    for c in channels:
        name = names.get(c.slot, c.slot)
        if name.lower() in seen or name.lower() == "status":
            unique = f"{name} [{c.slot}]"
            if len(unique) > BDF_LABEL_MAX:
                unique = f"{name[:BDF_LABEL_MAX - len(c.slot) - 3]} [{c.slot}]"
            notes.append(f"{c.slot} renamed to '{unique}' ('{name}' is already used)")
            name = unique
        seen.add(name.lower())
        out.append(replace(c, label=name))
    return out, notes
