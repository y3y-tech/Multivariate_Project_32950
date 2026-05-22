"""PCA -> Gaussian Mixture Model pipeline on the Communities and Crime dataset.

Course: STAT 24620 / FINM 34700 / STAT 32950 -- Multivariate Statistical Analysis.
Unsupervised structure discovery: cluster communities on socioeconomic features
with the crime outcome held out, then bring crime back as a validation variable.

Run from anywhere:
    python gmm_communities_crime.py

All randomness is seeded by RANDOM_STATE; all artifacts are written to OUTPUT_DIR.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless backend so figures are written directly to disk
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = (SCRIPT_DIR / ".." / "Data" / "communities_crime.csv").resolve()
OUTPUT_DIR = SCRIPT_DIR / "outputs"

OUTCOME_CONTINUOUS = "ViolentCrimesPerPop"
OUTCOME_BINARY = "HighViolentCrime"

VARIANCE_THRESHOLD = 0.85          # smallest n_pcs reaching this cumulative explained variance
K_GRID = list(range(1, 9))          # 1..8 mixture components
COV_TYPES = ["spherical", "diag", "tied", "full"]
REG_COVAR_DEFAULT = 1e-6
REG_COVAR_FALLBACK = 1e-4
N_INIT = 10
MAX_ITER = 500

TOP_FEATURES_PER_CLUSTER = 8        # for table_cluster_top_features.csv
HEATMAP_N_FEATURES = 15             # features shown in fig3 heatmap


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def load_data(input_path: Path, log_lines: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load CSV, coerce HighViolentCrime to 0/1, validate, split off outcomes."""
    df = pd.read_csv(input_path)

    # The provided file encodes HighViolentCrime as the strings "High"/"Low".
    # The spec calls for integer 0/1; map first, then run the numeric assertion.
    if df[OUTCOME_BINARY].dtype == object or pd.api.types.is_string_dtype(df[OUTCOME_BINARY]):
        mapping = {"High": 1, "Low": 0}
        unknown = set(df[OUTCOME_BINARY].unique()) - set(mapping)
        if unknown:
            raise ValueError(f"Unexpected values in {OUTCOME_BINARY}: {unknown}")
        df[OUTCOME_BINARY] = df[OUTCOME_BINARY].map(mapping).astype(int)
    else:
        df[OUTCOME_BINARY] = df[OUTCOME_BINARY].astype(int)

    if df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Input contains NaN values in columns: {bad}")

    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric columns present after coercion: {non_numeric}")

    for required in (OUTCOME_CONTINUOUS, OUTCOME_BINARY):
        if required not in df.columns:
            raise ValueError(f"Required outcome column missing: {required}")

    y_cont = df[OUTCOME_CONTINUOUS].astype(float)
    y_bin = df[OUTCOME_BINARY].astype(int)
    X = df.drop(columns=[OUTCOME_CONTINUOUS, OUTCOME_BINARY])

    # Drop zero-variance feature columns: constant features add no information
    # and would yield zero scale during standardization.
    zero_var_mask = X.var(axis=0) == 0
    zero_var_cols = X.columns[zero_var_mask].tolist()
    if zero_var_cols:
        X = X.drop(columns=zero_var_cols)

    log_lines.append(f"Rows: {len(df)}")
    log_lines.append(f"Feature columns used: {X.shape[1]}")
    log_lines.append(f"Zero-variance columns dropped: {len(zero_var_cols)}")
    if zero_var_cols:
        log_lines.append(f"  Names: {zero_var_cols}")
    log_lines.append(f"Outcome (continuous): {OUTCOME_CONTINUOUS}  range=[{y_cont.min():.4f}, {y_cont.max():.4f}]")
    log_lines.append(f"Outcome (binary):     {OUTCOME_BINARY}  positives={int(y_bin.sum())}/{len(y_bin)}")
    return X, y_cont, y_bin


def run_pca(
    X: pd.DataFrame,
    output_dir: Path,
    log_lines: list[str],
) -> tuple[np.ndarray, PCA, int]:
    """Standardize, fit PCA, choose n_pcs by cumulative variance, save scree."""
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    pca_full = PCA(random_state=RANDOM_STATE).fit(X_std)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n_pcs = int(np.searchsorted(cum, VARIANCE_THRESHOLD) + 1)
    n_pcs = max(1, min(n_pcs, len(cum)))

    pca = PCA(n_components=n_pcs, random_state=RANDOM_STATE).fit(X_std)
    X_pca = pca.transform(X_std)

    log_lines.append(
        f"PCA: n_pcs={n_pcs} reaches cumulative variance {cum[n_pcs - 1]:.4f} "
        f"(threshold {VARIANCE_THRESHOLD})"
    )

    # Scree / cumulative-variance figure.
    fig, ax1 = plt.subplots(figsize=(9, 5))
    comp_idx = np.arange(1, len(cum) + 1)
    ax1.bar(comp_idx, pca_full.explained_variance_ratio_, color="steelblue", alpha=0.7,
            label="Per-component variance")
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance ratio", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(comp_idx, cum, color="darkorange", marker="o", markersize=3,
             label="Cumulative variance")
    ax2.axhline(VARIANCE_THRESHOLD, color="red", linestyle="--", linewidth=1,
                label=f"Threshold {VARIANCE_THRESHOLD}")
    ax2.axvline(n_pcs, color="green", linestyle=":", linewidth=1.5,
                label=f"Chosen n_pcs={n_pcs}")
    ax2.set_ylabel("Cumulative explained variance", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax2.set_ylim(0, 1.02)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    plt.title("PCA scree and cumulative explained variance")
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_pca_scree.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return X_pca, pca, n_pcs


def _fit_gmm(X_pca: np.ndarray, K: int, cov: str, reg_covar: float) -> GaussianMixture:
    return GaussianMixture(
        n_components=K,
        covariance_type=cov,
        n_init=N_INIT,
        init_params="kmeans",       # k-means initialization (Lecture 6 recommendation)
        reg_covar=reg_covar,        # ridge on covariance to avoid singular components
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    ).fit(X_pca)


def select_gmm(
    X_pca: np.ndarray,
    output_dir: Path,
    log_lines: list[str],
) -> tuple[int, str, pd.DataFrame]:
    """Joint BIC sweep over (K, covariance_type). Returns the winning pair."""
    bic_grid = pd.DataFrame(index=K_GRID, columns=COV_TYPES, dtype=float)
    bic_grid.index.name = "K"
    reg_used: dict[tuple[int, str], float] = {}

    cov_singular_retry: set[str] = set()

    for cov in COV_TYPES:
        for K in K_GRID:
            reg_covar = REG_COVAR_DEFAULT
            try:
                gmm = _fit_gmm(X_pca, K, cov, reg_covar)
                bic = gmm.bic(X_pca)
                if not np.isfinite(bic):
                    raise ValueError(f"Non-finite BIC ({bic})")
            except Exception as exc:  # numerical / convergence failure
                log_lines.append(f"  GMM fit failed at (K={K}, cov={cov}, reg={reg_covar:g}): {exc!r}")
                if cov not in cov_singular_retry:
                    cov_singular_retry.add(cov)
                    reg_covar = REG_COVAR_FALLBACK
                    log_lines.append(f"  Retrying cov={cov} once with reg_covar={reg_covar:g}")
                    try:
                        gmm = _fit_gmm(X_pca, K, cov, reg_covar)
                        bic = gmm.bic(X_pca)
                        if not np.isfinite(bic):
                            raise ValueError(f"Non-finite BIC ({bic})")
                    except Exception as exc2:
                        log_lines.append(f"  Retry also failed at (K={K}, cov={cov}): {exc2!r}")
                        bic = np.nan
                else:
                    bic = np.nan
            bic_grid.loc[K, cov] = bic
            reg_used[(K, cov)] = reg_covar

    bic_grid.to_csv(output_dir / "table_bic_grid.csv")

    # Pick the lowest finite BIC.
    if bic_grid.notna().any().any():
        flat = bic_grid.stack()
        best_K, best_cov = flat.idxmin()
        best_bic = flat.min()
    else:
        raise RuntimeError("All GMM fits failed; no BIC available for selection.")

    log_lines.append("BIC grid (rows=K, cols=covariance_type):")
    log_lines.append(bic_grid.to_string(float_format=lambda v: f"{v:.2f}"))
    log_lines.append(f"Selected K*={best_K}, cov*={best_cov} with BIC={best_bic:.2f}")
    if best_K == 1:
        log_lines.append("NOTE: K*==1 selected; BIC prefers a single Gaussian (no multi-cluster structure).")

    # Figure 2: BIC vs K, one line per covariance type.
    fig, ax = plt.subplots(figsize=(8, 5))
    for cov in COV_TYPES:
        ax.plot(bic_grid.index, bic_grid[cov], marker="o", label=cov)
    ax.scatter([best_K], [best_bic], s=140, facecolors="none", edgecolors="red",
               linewidths=2, zorder=5, label=f"min: K={best_K}, {best_cov}")
    ax.set_xlabel("Number of components K")
    ax.set_ylabel("BIC (lower is better)")
    ax.set_title("Gaussian Mixture BIC sweep over K and covariance type")
    ax.set_xticks(K_GRID)
    ax.grid(True, alpha=0.3)
    ax.legend(title="covariance_type", loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_bic_by_K_and_cov.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return int(best_K), str(best_cov), bic_grid


def fit_final(
    X_pca: np.ndarray,
    K: int,
    cov: str,
    log_lines: list[str],
) -> tuple[GaussianMixture, np.ndarray, np.ndarray, np.ndarray]:
    """Refit the winning GMM on all of X_pca and return responsibilities + labels."""
    # Use the default reg_covar; if it fails for the winner, fall back once.
    try:
        gmm = _fit_gmm(X_pca, K, cov, REG_COVAR_DEFAULT)
    except Exception as exc:
        log_lines.append(f"Final fit failed at default reg_covar: {exc!r}. Retrying with {REG_COVAR_FALLBACK:g}.")
        gmm = _fit_gmm(X_pca, K, cov, REG_COVAR_FALLBACK)

    resp = gmm.predict_proba(X_pca)
    hard = gmm.predict(X_pca)
    max_resp = resp.max(axis=1)

    counts = pd.Series(hard).value_counts().sort_index()
    log_lines.append("Cluster sizes (hard assignment):")
    for cluster_id, n in counts.items():
        log_lines.append(f"  cluster {cluster_id}: {int(n)}")
    log_lines.append(f"Average max responsibility: {max_resp.mean():.4f} (1/K = {1.0 / K:.4f})")

    return gmm, resp, hard, max_resp


def profile_clusters(
    X_std_df: pd.DataFrame,
    hard: np.ndarray,
    output_dir: Path,
    log_lines: list[str],
) -> pd.DataFrame:
    """Per-cluster means in standardized-original-feature space; heatmap + tables."""
    df = X_std_df.copy()
    df["_cluster"] = hard
    profiles = df.groupby("_cluster").mean()
    profiles.index.name = "cluster"
    profiles.to_csv(output_dir / "table_cluster_profiles.csv")

    # Top features per cluster by absolute deviation from 0 (i.e. from the overall mean).
    rows = []
    for cluster_id, row in profiles.iterrows():
        top = row.abs().sort_values(ascending=False).head(TOP_FEATURES_PER_CLUSTER)
        for rank, (feat, abs_val) in enumerate(top.items(), start=1):
            rows.append({
                "cluster": cluster_id,
                "rank": rank,
                "feature": feat,
                "standardized_mean": row[feat],
            })
    top_table = pd.DataFrame(rows)
    top_table.to_csv(output_dir / "table_cluster_top_features.csv", index=False)

    # Heatmap: features with the largest spread of cluster means.
    if profiles.shape[0] >= 2:
        spread = profiles.var(axis=0, ddof=0).sort_values(ascending=False)
        chosen = spread.head(HEATMAP_N_FEATURES).index.tolist()
    else:
        # K* == 1: pick features that are most far from zero across the single cluster.
        chosen = profiles.iloc[0].abs().sort_values(ascending=False).head(HEATMAP_N_FEATURES).index.tolist()

    heatmap_df = profiles[chosen]
    vmax = float(np.nanmax(np.abs(heatmap_df.values))) if heatmap_df.size else 1.0
    vmax = max(vmax, 1e-6)

    fig_h = max(3.5, 0.55 * heatmap_df.shape[0] + 2.5)
    fig_w = max(8.0, 0.55 * heatmap_df.shape[1] + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        heatmap_df,
        cmap="RdBu_r",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Standardized feature mean (SD units from overall mean)"},
        linewidths=0.3,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("Cluster profiles in standardized feature space (top-spread features)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Cluster")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_cluster_profile_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return profiles


def crime_comparison(
    y_cont: pd.Series,
    y_bin: pd.Series,
    hard: np.ndarray,
    output_dir: Path,
    log_lines: list[str],
) -> float:
    """Held-out crime variable vs. clusters: boxplot, summary table, crosstab, ARI."""
    cluster_series = pd.Series(hard, index=y_cont.index, name="cluster")
    combo = pd.DataFrame({
        "cluster": cluster_series,
        OUTCOME_CONTINUOUS: y_cont.values,
        OUTCOME_BINARY: y_bin.values,
    })

    # Boxplot: crime distribution by cluster.
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=combo,
        x="cluster",
        y=OUTCOME_CONTINUOUS,
        ax=ax,
        hue="cluster",
        palette="Set2",
        legend=False,
    )
    sns.stripplot(
        data=combo,
        x="cluster",
        y=OUTCOME_CONTINUOUS,
        ax=ax,
        color="black",
        size=1.5,
        alpha=0.3,
        jitter=0.25,
    )
    ax.set_title(f"{OUTCOME_CONTINUOUS} by GMM hard cluster (held out of fitting)")
    ax.set_xlabel("Cluster")
    ax.set_ylabel(OUTCOME_CONTINUOUS)
    fig.tight_layout()
    fig.savefig(output_dir / "fig4_crime_by_cluster_boxplot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Per-cluster summary.
    summary = combo.groupby("cluster").agg(
        count=(OUTCOME_CONTINUOUS, "size"),
        mean_violent=(OUTCOME_CONTINUOUS, "mean"),
        median_violent=(OUTCOME_CONTINUOUS, "median"),
        high_crime_rate=(OUTCOME_BINARY, "mean"),
    )
    summary.to_csv(output_dir / "table_crime_by_cluster.csv")

    # Crosstab: counts + row-normalized proportions side by side.
    counts_ct = pd.crosstab(combo["cluster"], combo[OUTCOME_BINARY])
    counts_ct.columns = [f"count_HighViolentCrime={c}" for c in counts_ct.columns]
    props_ct = pd.crosstab(combo["cluster"], combo[OUTCOME_BINARY], normalize="index")
    props_ct.columns = [f"prop_HighViolentCrime={c}" for c in props_ct.columns]
    crosstab = pd.concat([counts_ct, props_ct], axis=1)
    crosstab.to_csv(output_dir / "table_cluster_vs_highcrime.csv")

    ari = adjusted_rand_score(y_bin.values, hard)

    log_lines.append("Per-cluster crime summary:")
    log_lines.append(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    log_lines.append("Cluster x HighViolentCrime crosstab (counts + row-normalized proportions):")
    log_lines.append(crosstab.to_string(float_format=lambda v: f"{v:.4f}"))
    log_lines.append(f"Adjusted Rand Index (cluster vs HighViolentCrime): {ari:.4f}")

    return ari


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    sns.set_theme(style="whitegrid")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    log_lines.append("=== GMM on Communities and Crime ===")
    log_lines.append(f"RANDOM_STATE = {RANDOM_STATE}")
    log_lines.append(f"Input:  {INPUT_PATH}")
    log_lines.append(f"Output: {OUTPUT_DIR}")
    log_lines.append("")

    log_lines.append("--- Step 1: load + held-out outcome split ---")
    X, y_cont, y_bin = load_data(INPUT_PATH, log_lines)
    log_lines.append("")

    # Keep a labeled standardized DataFrame for downstream profiling.
    scaler = StandardScaler()
    X_std_arr = scaler.fit_transform(X)
    X_std_df = pd.DataFrame(X_std_arr, columns=X.columns, index=X.index)

    log_lines.append("--- Step 3: PCA ---")
    X_pca, _pca, n_pcs = run_pca(X, OUTPUT_DIR, log_lines)
    log_lines.append("")

    log_lines.append("--- Step 4: BIC sweep over K x covariance_type ---")
    best_K, best_cov, _bic_grid = select_gmm(X_pca, OUTPUT_DIR, log_lines)
    log_lines.append("")

    log_lines.append("--- Step 5: fit final GMM on all data ---")
    gmm, resp, hard, max_resp = fit_final(X_pca, best_K, best_cov, log_lines)
    log_lines.append("")

    log_lines.append("--- Step 6: cluster profiles in standardized feature space ---")
    profile_clusters(X_std_df, hard, OUTPUT_DIR, log_lines)
    log_lines.append("")

    log_lines.append("--- Step 7: crime comparison (held-out outcome) ---")
    crime_comparison(y_cont, y_bin, hard, OUTPUT_DIR, log_lines)
    log_lines.append("")

    log_text = "\n".join(str(line) for line in log_lines)
    (OUTPUT_DIR / "results_log.txt").write_text(log_text + "\n")
    print(log_text)


if __name__ == "__main__":
    main()
