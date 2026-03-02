"""
visualize.py — Génération des courbes Privacy vs Utility.

Produit :
  1. Re-identification rate vs k (par nombre de QI)
  2. ML Accuracy loss vs k (par nombre de QI)
  3. Privacy vs Utility trade-off (scatter)
  4. Heatmap re-identification (k × n_qi)
  5. Résumé complet en un seul figure multi-panneaux
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap


# Palette de couleurs cohérente
COLORS = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6",
          "#1abc9c", "#e84393"]


def load_results(results_dir: str = "results") -> pd.DataFrame:
    """Charge le CSV de résultats."""
    csv_path = os.path.join(results_dir, "benchmark_summary.csv")
    df = pd.read_csv(csv_path)
    return df[df["status"] == "success"]


def plot_reid_vs_k(df: pd.DataFrame, output_dir: str = "results"):
    """
    Graphe 1 : Taux de ré-identification en fonction de k,
    une courbe par nombre de QI.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (n_qi, group) in enumerate(df.groupby("n_qi")):
        group = group.sort_values("k")
        ax.plot(group["k"], group["no_match_rate"] * 100,
                marker="o", linewidth=2, color=COLORS[i % len(COLORS)],
                label=f"{n_qi} QI", markersize=8)

    ax.set_xlabel("k (k-anonymity)", fontsize=13)
    ax.set_ylabel("No-Match Rate (% cibles introuvables)", fontsize=13)
    ax.set_title("Linkage Attack : Protection Strength vs k", fontsize=14,
                 fontweight="bold")
    ax.legend(title="Nb de QI connus\npar l'attaquant", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-2, 105)

    plt.tight_layout()
    path = os.path.join(output_dir, "plot_reid_vs_k.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  [PLOT] {path}")


def plot_ml_loss_vs_k(df: pd.DataFrame, output_dir: str = "results"):
    """
    Graphe 2 : Perte d'accuracy ML en fonction de k.
    """
    model_col = None
    for candidate in ["GradientBoosting_acc_loss", "RandomForest_acc_loss"]:
        if candidate in df.columns and df[candidate].notna().any():
            model_col = candidate
            break

    if model_col is None:
        print("  [SKIP] Pas de données ML disponibles")
        return

    model_name = model_col.replace("_acc_loss", "")

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (n_qi, group) in enumerate(df.groupby("n_qi")):
        group = group.sort_values("k")
        vals = group[model_col].dropna()
        if vals.empty:
            continue
        ax.plot(group["k"], group[model_col] * 100,
                marker="s", linewidth=2, color=COLORS[i % len(COLORS)],
                label=f"{n_qi} QI", markersize=8)

    ax.set_xlabel("k (k-anonymity)", fontsize=13)
    ax.set_ylabel("Perte d'accuracy (%)", fontsize=13)
    ax.set_title(f"ML Utility Loss ({model_name}) vs k", fontsize=14,
                 fontweight="bold")
    ax.legend(title="Nb de QI", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(output_dir, "plot_ml_loss_vs_k.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  [PLOT] {path}")


def plot_privacy_vs_utility(df: pd.DataFrame, output_dir: str = "results"):
    """
    Graphe 3 : Scatter Privacy (1 - re_id_rate) vs Utility (1 - info_loss).
    Chaque point est une config (k, n_qi). Couleur = k, taille = n_qi.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Use no_match_rate as privacy metric (more informative than re_id_rate
    # since k-anonymity guarantees 0% re-identification)
    privacy = df["no_match_rate"] * 100
    utility = (1 - df["mean_js"]) * 100

    scatter = ax.scatter(utility, privacy,
                         c=df["k"], cmap="RdYlGn", s=df["n_qi"] * 40,
                         edgecolors="black", linewidth=0.5, alpha=0.8)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("k (k-anonymity)", fontsize=12)

    # Annoter quelques points
    for _, row in df.iterrows():
        ax.annotate(f"k={int(row['k'])}\n{int(row['n_qi'])}QI",
                    (((1 - row["mean_js"]) * 100),
                     (row["no_match_rate"] * 100)),
                    fontsize=6, alpha=0.7, ha="center", va="bottom")

    ax.set_xlabel("Utility Score (1 - JS Divergence) %", fontsize=13)
    ax.set_ylabel("Privacy Score (No-Match Rate) %", fontsize=13)
    ax.set_title("Trade-off Privacy vs Utility", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "plot_privacy_vs_utility.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  [PLOT] {path}")


def plot_heatmap_reid(df: pd.DataFrame, output_dir: str = "results"):
    """
    Graphe 4 : Heatmap du taux de ré-identification (k × n_qi).
    """
    pivot = df.pivot_table(values="mia_accuracy", index="n_qi",
                           columns="k", aggfunc="first")
    pivot = pivot.sort_index(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = LinearSegmentedColormap.from_list("privacy",
                                              ["#2ecc71", "#f1c40f", "#e74c3c"])
    im = ax.imshow(pivot.values * 100, cmap=cmap, aspect="auto",
                    vmin=40, vmax=100)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"k={int(c)}" for c in pivot.columns], fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{int(n)} QI" for n in pivot.index], fontsize=11)

    # Valeurs dans les cellules
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = "white" if val * 100 > 70 else "black"
                ax.text(j, i, f"{val*100:.0f}%", ha="center", va="center",
                        fontsize=12, fontweight="bold", color=color)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("MIA Accuracy (%)", fontsize=12)

    ax.set_xlabel("Force d'anonymisation (k)", fontsize=13)
    ax.set_ylabel("Connaissance de l'attaquant (nb QI)", fontsize=13)
    ax.set_title("Heatmap : Membership Inference Attack Accuracy", fontsize=14,
                 fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "plot_heatmap_reid.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  [PLOT] {path}")


def plot_mia_results(df: pd.DataFrame, output_dir: str = "results"):
    """
    Graphe 5 : Résultats de l'attaque MIA (accuracy) vs k.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (n_qi, group) in enumerate(df.groupby("n_qi")):
        group = group.sort_values("k")
        if "mia_accuracy" in group.columns and group["mia_accuracy"].notna().any():
            ax.plot(group["k"], group["mia_accuracy"] * 100,
                    marker="^", linewidth=2, color=COLORS[i % len(COLORS)],
                    label=f"{n_qi} QI", markersize=8)

    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.7,
               label="Hasard (50%)")
    ax.set_xlabel("k (k-anonymity)", fontsize=13)
    ax.set_ylabel("MIA Accuracy (%)", fontsize=13)
    ax.set_title("Membership Inference Attack : Accuracy vs k", fontsize=14,
                 fontweight="bold")
    ax.legend(title="Nb de QI", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(40, 105)

    plt.tight_layout()
    path = os.path.join(output_dir, "plot_mia_vs_k.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  [PLOT] {path}")


def plot_cost_vs_privacy(df: pd.DataFrame, output_dir: str = "results"):
    """
    Graphe 6 : Coût d'anonymisation vs protection obtenue.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Coût vs k
    for i, (n_qi, group) in enumerate(df.groupby("n_qi")):
        group = group.sort_values("k")
        ax1.plot(group["k"], group["cost"],
                 marker="D", linewidth=2, color=COLORS[i % len(COLORS)],
                 label=f"{n_qi} QI", markersize=8)

    ax1.set_xlabel("k (k-anonymity)", fontsize=12)
    ax1.set_ylabel("Coût (somme niveaux de généralisation)", fontsize=12)
    ax1.set_title("Coût d'anonymisation vs k", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Suppression rate vs k
    for i, (n_qi, group) in enumerate(df.groupby("n_qi")):
        group = group.sort_values("k")
        ax2.plot(group["k"], group["suppression_rate"] * 100,
                 marker="v", linewidth=2, color=COLORS[i % len(COLORS)],
                 label=f"{n_qi} QI", markersize=8)

    ax2.set_xlabel("k (k-anonymity)", fontsize=12)
    ax2.set_ylabel("Taux de suppression (%)", fontsize=12)
    ax2.set_title("Données supprimées vs k", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "plot_cost_analysis.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  [PLOT] {path}")


def generate_all_plots(results_dir: str = "results"):
    """Génère tous les graphiques à partir du CSV de résultats."""
    print("\n[VISUALIZE] Génération des graphiques...")
    df = load_results(results_dir)

    if df.empty:
        print("  [ERREUR] Aucun résultat trouvé.")
        return

    plot_reid_vs_k(df, results_dir)
    plot_ml_loss_vs_k(df, results_dir)
    plot_privacy_vs_utility(df, results_dir)
    plot_heatmap_reid(df, results_dir)
    plot_mia_results(df, results_dir)
    plot_cost_vs_privacy(df, results_dir)

    print("[VISUALIZE] Tous les graphiques générés ✓")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Génère les plots du benchmark")
    parser.add_argument("--results", type=str, default="results",
                        help="Dossier contenant benchmark_summary.csv")
    args = parser.parse_args()
    generate_all_plots(args.results)
