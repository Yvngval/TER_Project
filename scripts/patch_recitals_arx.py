#!/usr/bin/env python3
"""Patch the RECITALS ARX adapter so suppression_limit (%) is converted to [0,1].

This project stores ``suppression_limit`` as an integer percentage (e.g. 10 for 10%),
while ARX expects a ratio between 0 and 1. The upstream adapter currently forwards the
integer directly to ``configuration.setSuppressionLimit(...)``, which causes:

    java.lang.IllegalArgumentException: Suppression rate 10.0 must be in [0, 1]

This script patches the installed/local RECITALS source by replacing:

    configuration.setSuppressionLimit(config.suppression_limit)

with:

    configuration.setSuppressionLimit(config.suppression_limit / 100.0)

Usage
-----
From TER_Project:
    python scripts/patch_recitals_arx.py

Optional explicit path:
    python scripts/patch_recitals_arx.py --recitals-root ../RECITALS-anonymization-manager
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

TARGET_RELATIVE = Path("src/anonymization_manager/adapters/arx/arx.py")
OLD = "configuration.setSuppressionLimit(config.suppression_limit)"
NEW = "configuration.setSuppressionLimit(config.suppression_limit / 100.0)"


def candidate_roots(script_dir: Path) -> list[Path]:
    project_root = script_dir.parent
    parent = project_root.parent
    return [
        parent / "RECITALS-anonymization-manager",
        project_root / "RECITALS-anonymization-manager",
        Path.cwd() / "RECITALS-anonymization-manager",
        Path.cwd().parent / "RECITALS-anonymization-manager",
    ]


def resolve_recitals_root(explicit: str | None, script_dir: Path) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        target = root / TARGET_RELATIVE
        if not target.exists():
            raise FileNotFoundError(
                f"Le fichier cible est introuvable ici : {target}\n"
                "Passe le bon chemin avec --recitals-root."
            )
        return root

    for root in candidate_roots(script_dir):
        target = root / TARGET_RELATIVE
        if target.exists():
            return root.resolve()

    tried = "\n - ".join(str(p.resolve()) for p in candidate_roots(script_dir))
    raise FileNotFoundError(
        "Impossible de trouver automatiquement le dépôt RECITALS.\n"
        f"Chemins testés :\n - {tried}\n"
        "Relance avec --recitals-root /chemin/vers/RECITALS-anonymization-manager"
    )


def patch_file(target: Path) -> str:
    text = target.read_text(encoding="utf-8")

    if NEW in text:
        return "already_patched"

    if OLD not in text:
        raise RuntimeError(
            "La ligne attendue n'a pas été trouvée dans arx.py.\n"
            "Le fichier a peut-être changé ; vérifie manuellement."
        )

    backup = target.with_suffix(target.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(target, backup)

    target.write_text(text.replace(OLD, NEW), encoding="utf-8")
    return str(backup)


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch suppression_limit for ARX in RECITALS.")
    parser.add_argument(
        "--recitals-root",
        help="Chemin vers le dépôt RECITALS-anonymization-manager.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    try:
        root = resolve_recitals_root(args.recitals_root, script_dir)
        target = root / TARGET_RELATIVE
        result = patch_file(target)
    except Exception as exc:
        print(f"[ERREUR] {exc}", file=sys.stderr)
        return 1

    print(f"[OK] Dépôt RECITALS trouvé : {root}")
    print(f"[OK] Fichier patché : {target}")
    if result == "already_patched":
        print("[INFO] Le patch était déjà appliqué.")
    else:
        print(f"[OK] Sauvegarde créée : {result}")
    print("[OK] Garde suppression_limit comme entier (ex: 10 pour 10%) dans tes JSON TER.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
