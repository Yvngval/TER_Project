#!/usr/bin/env python3
"""
run_benchmark.py — Script principal pour lancer le benchmark complet.

Usage :
    python run_benchmark.py                   # Benchmark complet
    python run_benchmark.py --quick           # Mode rapide (test)
    python run_benchmark.py --no-ml           # Sans benchmark ML
    python run_benchmark.py --plots-only      # Regénérer les plots seulement
"""

import sys
import os

# S'assurer que le répertoire courant est dans le path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark import run_full_benchmark
from visualize import generate_all_plots


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline complet : Benchmark Privacy vs Utility sur Adult dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python run_benchmark.py --data data/adult.csv
  python run_benchmark.py --quick --no-ml
  python run_benchmark.py --plots-only
        """
    )
    parser.add_argument("--data", type=str, default="data/adult.csv",
                        help="Chemin vers adult.csv (défaut: data/adult.csv)")
    parser.add_argument("--output", type=str, default="results",
                        help="Dossier de sortie (défaut: results)")
    parser.add_argument("--no-ml", action="store_true",
                        help="Désactiver le benchmark ML (beaucoup plus rapide)")
    parser.add_argument("--quick", action="store_true",
                        help="Mode rapide : moins de configs testées")
    parser.add_argument("--plots-only", action="store_true",
                        help="Regénérer les plots depuis les résultats existants")

    args = parser.parse_args()

    # ── Mode plots seulement ──────────────────────────────
    if args.plots_only:
        generate_all_plots(args.output)
        return

    # ── Mode rapide ───────────────────────────────────────
    if args.quick:
        import config
        config.K_VALUES = [2, 5, 10]
        config.QI_SUBSETS = [
            ["age", "sex"],
            ["age", "sex", "race", "marital-status", "education"],
        ]
        config.N_TARGETS = 50

    # ── Vérifier que le fichier existe ────────────────────
    if not os.path.isfile(args.data):
        print(f"[ERREUR] Fichier introuvable : {args.data}")
        print(f"  Placez adult.csv dans le dossier data/ ou spécifiez --data")
        sys.exit(1)

    # ── Lancer le benchmark ───────────────────────────────
    results = run_full_benchmark(
        data_filepath=args.data,
        output_dir=args.output,
        run_ml=not args.no_ml
    )

    # ── Générer les graphiques ────────────────────────────
    generate_all_plots(args.output)

    print(f"\n✅ Pipeline terminé. Résultats dans : {args.output}/")
    print(f"   - benchmark_results.json  (détails complets)")
    print(f"   - benchmark_summary.csv   (tableau résumé)")
    print(f"   - plot_*.png              (graphiques)")


if __name__ == "__main__":
    main()
