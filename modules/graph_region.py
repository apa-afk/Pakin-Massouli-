# =========================
# Imports (unique list)
# =========================
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess

# =========================
# Small helpers
# =========================
def _col_exists(df, col):
    return col in df.columns

def _get_first_existing(df, candidates, name="column"):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the candidate {name}s exist: {candidates}")

def _safe_log1p(x):
    return np.log1p(pd.to_numeric(x, errors="coerce"))

def _richesse_col_for_year(df, year):
    # richesse available up to 2023 in your data; map 2024 -> 2023 by default
    y = min(int(year), 2023)
    return f"richesse_{y}"

def _foyers_col_for_year(df, year):
    y = min(int(year), 2023)
    return f"foyers_{y}"

def _sumacc_col(year, sex):
    return f"sum_acc_{int(year)}_{sex}"

def _avggrav_col(year, sex):
    return f"avg_grav_{int(year)}_{sex}"

def _avgan_col(year, sex):
    return f"avg_an_nais_{int(year)}_{sex}"


# ============================================================
# Scatter dynamique : Richesse × Accidents (trajectoires)
# ============================================================
def plot_richesse_accidents_trajectories(
    region_df,
    years=range(2020, 2025),
    sex="H",
    normalize_by="foyers",  # "none" or "foyers"
    title=None,
    annotate=True
):
    """
    Dynamic scatter with trajectories over time:
      x = log(richesse_t)
      y = log(accidents_t) or log(accidents_t / foyers_t)
      lines connect years for each region.

    region_df must contain: region_id, region, richesse_YYYY, foyers_YYYY, sum_acc_YYYY_{sex}
    """
    df = region_df.copy()

    fig, ax = plt.subplots(figsize=(10, 7))

    for _, r in df.iterrows():
        xs, ys = [], []
        for y in years:
            col_rich = _richesse_col_for_year(df, y)
            col_sum = _sumacc_col(y, sex)

            if not _col_exists(df, col_rich) or not _col_exists(df, col_sum):
                continue

            x = r[col_rich]
            acc = r[col_sum]

            if normalize_by == "foyers":
                col_f = _foyers_col_for_year(df, y)
                if not _col_exists(df, col_f):
                    continue
                denom = r[col_f]
                acc = (pd.to_numeric(acc, errors="coerce") / pd.to_numeric(denom, errors="coerce"))

            xs.append(np.log1p(pd.to_numeric(x, errors="coerce")))
            ys.append(np.log1p(pd.to_numeric(acc, errors="coerce")))

        if len(xs) >= 2:
            ax.plot(xs, ys, marker="o", linewidth=1)
            if annotate:
                ax.text(xs[-1], ys[-1], str(r["region_id"]), fontsize=9)

    ax.set_xlabel("log(1 + richesse)")
    ax.set_ylabel(f"log(1 + accidents{' / foyers' if normalize_by=='foyers' else ''})")
    ax.set_title(title or f"Trajectoires Richesse ↔ Accidents ({sex}), {min(years)}–{max(years)}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==========================================
# Diff-in-Diff visuel : choc Covid 2020
# ==========================================
def plot_did_covid(
    region_df,
    outcome="sum_acc",       # "sum_acc" or "avg_grav" or "avg_an_nais"
    sex="H",
    years=range(2020, 2025),
    baseline_richesse_year=2019,
    treat_quantile=0.75,
    normalize_by="foyers",   # only applied if outcome == "sum_acc"
    title=None
):
    """
    Visual DiD: group regions by baseline richness (top quantile treated).
    Plots average outcome by group over years with a vertical line at 2020.

    Note: BAAC outcomes start at 2020 in your data; this is still useful as a
    'shock window' plot + group differential over 2020–2024.
    """
    df = region_df.copy()

    col_rich_base = _richesse_col_for_year(df, baseline_richesse_year)
    if col_rich_base not in df.columns:
        raise KeyError(f"Missing {col_rich_base}")

    thr = pd.to_numeric(df[col_rich_base], errors="coerce").quantile(treat_quantile)
    df["treated"] = (pd.to_numeric(df[col_rich_base], errors="coerce") >= thr)

    series = []
    for y in years:
        if outcome == "sum_acc":
            col = _sumacc_col(y, sex)
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce")

            if normalize_by == "foyers":
                col_f = _foyers_col_for_year(df, y)
                if col_f not in df.columns:
                    continue
                vals = vals / pd.to_numeric(df[col_f], errors="coerce")

        elif outcome == "avg_grav":
            col = _avggrav_col(y, sex)
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce")

        elif outcome == "avg_an_nais":
            col = _avgan_col(y, sex)
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce")

        else:
            raise ValueError("outcome must be one of: sum_acc, avg_grav, avg_an_nais")

        tmp = df[["treated"]].copy()
        tmp["year"] = y
        tmp["val"] = vals
        series.append(tmp)

    long = pd.concat(series, ignore_index=True)
    grp = long.groupby(["year", "treated"])["val"].mean().unstack()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(grp.index, grp[False], marker="o", label=f"Control (< Q{int(treat_quantile*100)})")
    ax.plot(grp.index, grp[True],  marker="o", label=f"Treated (≥ Q{int(treat_quantile*100)})")
    ax.axvline(2020, linestyle="--", linewidth=1)

    ylab = outcome
    if outcome == "sum_acc" and normalize_by == "foyers":
        ylab += " / foyers"

    ax.set_title(title or f"DiD visuel (groupes richesse {baseline_richesse_year}) — {ylab} ({sex})")
    ax.set_xlabel("Year")
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ==========================================================
# Alcool ↔ Accidents (H/F) + LOWESS (non-linéaire)
# ==========================================================
def plot_alcohol_accidents_lowess(
    region_df,
    alcohol_base="alc_bim",   # "alc_bim" or "alc_quo"
    alcohol_year=2021,
    outcome_year=2024,
    outcome="sum_acc",        # "sum_acc" or "avg_grav"
    normalize_by="foyers",    # only if outcome == "sum_acc"
    sexes=("H", "F"),
    title=None,
):
    """
    Faceted scatter (H/F) of alcohol vs accidents (or severity) with LOWESS curve.
    Uses alcohol column like f"{alcohol_base}_{sex}_{alcohol_year}".
    Uses outcome column like f"{outcome}_{outcome_year}_{sex}" where outcome in {"sum_acc","avg_grav"}.
    """
    df = region_df.copy()

    fig, axes = plt.subplots(1, len(sexes), figsize=(6.5 * len(sexes), 5), sharey=True)
    if len(sexes) == 1:
        axes = [axes]

    for ax, s in zip(axes, sexes):
        col_alc = f"{alcohol_base}_{s}_{alcohol_year}"
        if col_alc not in df.columns:
            raise KeyError(f"Missing alcohol column: {col_alc}")

        if outcome == "sum_acc":
            col_out = _sumacc_col(outcome_year, s)
            if col_out not in df.columns:
                raise KeyError(f"Missing outcome column: {col_out}")
            y = pd.to_numeric(df[col_out], errors="coerce")
            if normalize_by == "foyers":
                col_f = _foyers_col_for_year(df, outcome_year)
                if col_f not in df.columns:
                    raise KeyError(f"Missing foyers column: {col_f}")
                y = y / pd.to_numeric(df[col_f], errors="coerce")

        elif outcome == "avg_grav":
            col_out = _avggrav_col(outcome_year, s)
            if col_out not in df.columns:
                raise KeyError(f"Missing outcome column: {col_out}")
            y = pd.to_numeric(df[col_out], errors="coerce")

        else:
            raise ValueError("outcome must be 'sum_acc' or 'avg_grav'")

        x = pd.to_numeric(df[col_alc], errors="coerce")
        m = pd.DataFrame({"x": x, "y": y}).dropna()

        ax.scatter(m["x"], m["y"], alpha=0.85)

        # LOWESS (non-parametric)
        if len(m) >= 5:
            smoothed = lowess(m["y"], m["x"], frac=0.6, return_sorted=True)
            ax.plot(smoothed[:, 0], smoothed[:, 1], linewidth=2)

        ax.set_xlabel(f"{alcohol_base} ({s}) — {alcohol_year}")
        ax.set_title(f"{s}: {outcome} ({outcome_year})" + (f" / foyers" if outcome=="sum_acc" and normalize_by=="foyers" else ""))
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(outcome)
    fig.suptitle(title or f"Alcool ↔ {outcome} avec LOWESS (année alcool {alcohol_year})", y=1.02)
    plt.tight_layout()
    plt.show()


# ==========================================
# PCA + biplot (régions en projection)
# ==========================================
def plot_pca_biplot(
    region_df,
    features,                 # list of columns to use
    title="PCA biplot (regions)",
    label_col="region_id",
    standardize=True
):
    """
    PCA biplot (PC1 vs PC2) with loadings arrows.
    - features: list of numeric columns in region_df
    """
    df = region_df.copy()

    X = df[features].apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=0, how="any")
    df2 = df.loc[X.index].copy()

    if standardize:
        Xs = StandardScaler().fit_transform(X.values)
    else:
        Xs = X.values

    pca = PCA(n_components=2)
    Z = pca.fit_transform(Xs)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(Z[:, 0], Z[:, 1], alpha=0.9)

    # labels
    for i, lab in enumerate(df2[label_col].astype(str).values):
        ax.text(Z[i, 0], Z[i, 1], lab, fontsize=9)

    # loadings (arrows)
    # scale arrows to visible range
    loadings = pca.components_.T
    arrow_scale = 2.5 * np.max(np.abs(Z)) if np.max(np.abs(Z)) > 0 else 1.0

    for j, feat in enumerate(features):
        ax.arrow(0, 0, loadings[j, 0] * arrow_scale, loadings[j, 1] * arrow_scale,
                 head_width=0.04 * arrow_scale, length_includes_head=True)
        ax.text(loadings[j, 0] * arrow_scale * 1.08,
                loadings[j, 1] * arrow_scale * 1.08,
                feat, fontsize=9)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()

    return pca


# ======================================================
# Heatmap de corrélations conditionnelles (partielles)
# ======================================================
def plot_partial_corr_heatmap(
    region_df,
    variables,                # list of columns to show partial correlation among
    controls=None,            # list of control columns
    add_region_fe=False,      # include region fixed effects via dummies
    title="Partial correlation heatmap",
):
    """
    Partial correlation heatmap:
    - Residualize each variable v on controls (+ optional region FE),
      then compute correlation matrix of residuals.
    """
    df = region_df.copy()
    controls = controls or []

    # ---- Build design matrix X ----
    X_parts = []

    if controls:
        Xc = df[controls].apply(pd.to_numeric, errors="coerce")
        X_parts.append(Xc)

    if add_region_fe:
        # Make sure region_id is numeric int and dummies are float (not bool/object)
        reg = pd.to_numeric(df["region_id"], errors="coerce").astype("Int64")
        dummies = pd.get_dummies(reg, prefix="fe_region", drop_first=True).astype(float)
        X_parts.append(dummies)

    if not X_parts:
        raise ValueError("Provide controls and/or set add_region_fe=True for partial correlations.")

    X = pd.concat(X_parts, axis=1)

    # Force numeric float and add constant
    X = X.apply(pd.to_numeric, errors="coerce").astype(float)
    X = sm.add_constant(X, has_constant="add").astype(float)

    # ---- Residualize each variable ----
    resid_series = {}

    for v in variables:
        if v not in df.columns:
            raise KeyError(f"Missing variable column: {v}")

        y = pd.to_numeric(df[v], errors="coerce").astype(float)

        tmp = pd.concat([y.rename("y"), X], axis=1).dropna()
        y2 = tmp["y"].astype(float)
        X2 = tmp.drop(columns=["y"]).astype(float)

        # OLS on numeric arrays only
        model = sm.OLS(y2.to_numpy(), X2.to_numpy()).fit()
        yhat = model.predict(X2.to_numpy())
        resid_series[v] = pd.Series(y2.to_numpy() - yhat, index=tmp.index)

    resid_df = pd.concat(resid_series, axis=1).dropna()
    corr = resid_df.corr()

    # ---- Plot heatmap (matplotlib only) ----
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, aspect="auto")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)

    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

    return corr



# ==========================================================
# Coef plot économétrique (panel long + FE)
# ==========================================================
def plot_coef_panel_FE(
    region_df,
    years=range(2020, 2025),
    sexes=("H", "F"),
    alcohol_base="alc_bim",
    alcohol_year=2021,            # often only available for 2021 in your conso
    outcome="sum_acc",
    normalize_by_foyers=True,     # model on accidents per foyer (recommended)
    use_log=True,                 # log(1+y)
    title="Coefficient plot — FE region & year",
):
    """
    Builds a long panel (region_id, year, S) from region_df and runs FE regressions per sex:
      y ~ log(richesse_year) + alcohol_(sex, alcohol_year) + C(region_id) + C(year)

    Plots coefficient estimates with 95% CI for:
      - log_richesse
      - alcohol

    Returns dict of fitted models {sex: results}.
    """
    df = region_df.copy()

    rows = []
    for y in years:
        for s in sexes:
            col_sum = _sumacc_col(y, s)
            col_rich = _richesse_col_for_year(df, y)
            col_foy = _foyers_col_for_year(df, y)
            col_alc = f"{alcohol_base}_{s}_{alcohol_year}"

            needed = ["region_id", "region", col_sum, col_rich, col_alc]
            if normalize_by_foyers:
                needed.append(col_foy)

            missing = [c for c in needed if c not in df.columns]
            if missing:
                raise KeyError(f"Missing columns for year={y}, sex={s}: {missing}")

            tmp = df[needed].copy()
            tmp["year"] = int(y)
            tmp["S"] = s

            y_raw = pd.to_numeric(tmp[col_sum], errors="coerce")
            if normalize_by_foyers:
                denom = pd.to_numeric(tmp[col_foy], errors="coerce")
                y_raw = y_raw / denom

            tmp["y"] = y_raw
            tmp["richesse"] = pd.to_numeric(tmp[col_rich], errors="coerce")
            tmp["alcool"] = pd.to_numeric(tmp[col_alc], errors="coerce")

            rows.append(tmp[["region_id", "region", "year", "S", "y", "richesse", "alcool"]])

    panel = pd.concat(rows, ignore_index=True).dropna()

    # transforms
    if use_log:
        panel["y_t"] = np.log1p(panel["y"])
        panel["log_richesse"] = np.log1p(panel["richesse"])
    else:
        panel["y_t"] = panel["y"]
        panel["log_richesse"] = panel["richesse"]

    models = {}

    # Fit one model per sex (clean interpretation)
    coef_rows = []
    for s in sexes:
        sub = panel[panel["S"] == s].copy()

        # FE region + FE year
        # Note: alcohol is time-invariant here (2021 snapshot), so it will be identified via between-region variation.
        res = smf.ols("y_t ~ log_richesse + alcool + C(region_id) + C(year)", data=sub).fit(cov_type="HC3")
        models[s] = res

        for var in ["log_richesse", "alcool"]:
            est = res.params.get(var, np.nan)
            se = res.bse.get(var, np.nan)
            coef_rows.append({
                "S": s,
                "var": var,
                "coef": est,
                "low": est - 1.96 * se,
                "high": est + 1.96 * se,
            })

    coef_df = pd.DataFrame(coef_rows)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    x_positions = np.arange(len(coef_df))

    # arrange: group by sex then variable
    order = []
    for s in sexes:
        for v in ["log_richesse", "alcool"]:
            order.append((s, v))

    plot_df = pd.DataFrame(order, columns=["S", "var"]).merge(coef_df, on=["S", "var"], how="left")
    x = np.arange(len(plot_df))

    ax.errorbar(
        x, plot_df["coef"],
        yerr=[plot_df["coef"] - plot_df["low"], plot_df["high"] - plot_df["coef"]],
        fmt="o", capsize=4
    )

    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.S}:{r.var}" for r in plot_df.itertuples(index=False)], rotation=30, ha="right")
    ax.set_ylabel("Coefficient (95% CI)")
    ax.set_title(title + f"\nOutcome: {'log1p(acc/foyers)' if (outcome=='sum_acc' and normalize_by_foyers and use_log) else 'y'} | alcohol={alcohol_base}_{alcohol_year}")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()

    return models, panel
