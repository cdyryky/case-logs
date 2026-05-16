from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.constants import CONFIG_DIR, DEFAULT_CASE_CLASS
from app.utils import canonical_key


def visible_label(value: str) -> str:
    return " ".join(str(value).replace("\u00a0", " ").split())


def main() -> None:
    src = ROOT / "ACGME-class:area:type dictionary.txt"
    data = json.loads(src.read_text())
    seen = OrderedDict()
    for top_level_area, groups in data.items():
        area_label = visible_label(top_level_area)
        for group_name, entries in groups.items():
            type_label = visible_label(group_name)
            if area_label and type_label:
                seen[(DEFAULT_CASE_CLASS, area_label, type_label)] = None
            if not isinstance(entries, list):
                continue
            for entry in entries:
                area = entry.get("area")
                typ = entry.get("type")
                if not area or not typ:
                    continue
                key = (DEFAULT_CASE_CLASS, visible_label(area), visible_label(typ))
                seen[key] = None
    case_options = [
        {
            "class": {"visible_label": cls, "canonical_key": canonical_key(cls)},
            "area": {"visible_label": area, "canonical_key": canonical_key(area)},
            "type": {"visible_label": typ, "canonical_key": canonical_key(typ)},
        }
        for cls, area, typ in sorted(seen)
    ]
    payload = {
        "site": {
            "default": "University of California (Davis) Medical Center",
            "allowed": [
                "Kaiser Permanente Medical Center, Roseville",
                "Kaiser Permanente Medical Center, South Sacramento",
                "University of California (Davis) Medical Center",
            ],
        },
        "patient_type": {"allowed": ["Adult", "Pediatric"]},
        "role": {"allowed": ["Primary", "Secondary"], "avoid": ["Assisting"]},
        "case_year": {"allowed": [1, 2, 3, 4, 5]},
        "case_options": case_options,
    }
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (CONFIG_DIR / "acgme_dropdowns.yaml").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {CONFIG_DIR / 'acgme_dropdowns.yaml'} with {len(case_options)} options")


if __name__ == "__main__":
    main()
