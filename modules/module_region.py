import pandas as pd
import requests
import re
import numpy as np



def excel_fix(df):
    # header is on row 1
    df.columns = df.iloc[1].astype(str)

    # clean year column names (keep first column)
    cols = df.columns.tolist()
    cols[1:] = [str(int(float(c.replace("*", "").strip()))) for c in cols[1:]]
    df.columns = cols

    # keep data rows
    df = df.iloc[2:].reset_index(drop=True)
    df = df.iloc[:17].copy()

    # rename first column
    df = df.rename(columns={df.columns[0]: "region"})

    # normalize region names
    df["region"] = (
        df["region"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({
            "PROVENCE-ALPES-COTE -D'AZUR": "PROVENCE-ALPES-CÔTE D'AZUR",
            "AUVERGNE RHONE ALPES": "AUVERGNE-RHÔNE-ALPES",
            "CENTRE VAL DE LOIRE": "CENTRE-VAL DE LOIRE",
            "BOURGOGNE FRANCHE COMTE": "BOURGOGNE-FRANCHE-COMTÉ",
            "PAYS DE LA LOIRE": "PAYS DE LA LOIRE",
            "ILE-DE-FRANCE": "ÎLE-DE-FRANCE",
            "NOMBRE TOTAL": "TOTAL",
            "MONTANT TOTAL": "TOTAL",
        })
    )

    df = df[~df["region"].isin(["RÉSIDENTS ÉTRANGERS", "TOTAL"])].copy()

    # region -> INSEE reg code (same as df_conso.reg)
    region_to_reg = {
        "ÎLE-DE-FRANCE": 11,
        "CENTRE-VAL DE LOIRE": 24,
        "BOURGOGNE-FRANCHE-COMTÉ": 27,
        "NORMANDIE": 28,
        "HAUTS DE FRANCE": 32,
        "GRAND EST": 44,
        "PAYS DE LA LOIRE": 52,
        "BRETAGNE": 53,
        "NOUVELLE AQUITAINE": 75,
        "OCCITANIE": 76,
        "AUVERGNE-RHÔNE-ALPES": 84,
        "PROVENCE-ALPES-CÔTE D'AZUR": 93,
        "CORSE": 94,
        "OUTRE-MER": 1,
        # TOTAL and RÉSIDENTS ÉTRANGERS intentionally removed
    }

    df["region_id"] = df["region"].map(region_to_reg)

    # drop unmapped rows (safety)
    df = df[df["region_id"].notna()].copy()
    df["region_id"] = df["region_id"].astype(int)

    # reorder columns
    cols = ["region_id", "region"] + [
        c for c in df.columns if c not in ["region_id", "region"]
    ]
    return df[cols]



# actually used
def _prefix_year_cols(df, prefix):
    cols = {}
    for c in df.columns:
        m = re.search(r"(\d{4})", str(c))
        if m:
            y = m.group(1)
            cols[c] = f"{prefix}_{y}"
    return df.rename(columns=cols)



def prepare_alc(
    df,
    base,
    year_col="annee",
    reg_col="reg",
    region_col="reglib",
    sexe_col="sexe",
    outre_mer_regs=range(1, 5),
    outre_mer_name="Outre mer",
    outre_mer_reg_value=0,
    aggfunc="sum",
):
    df = df.copy()

    value_col = f"{base}_v"
    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' not found in dataframe")

    # 1) Keep only necessary columns
    df = df[[year_col, reg_col, region_col, sexe_col, value_col]].copy()

    # 2) Rename identifiers
    df = df.rename(
        columns={
            reg_col: "region_id",
            region_col: "region",
        }
    )

    # 3) Clean region names
    df["region"] = df["region"].astype(str).str.strip()
    df = df[df["region"].notna() & (df["region"] != "")]

    # 4) Map sexe → H/F/HF
    sexe_map = {
        "Hommes": "H",
        "Femmes": "F",
        "Hommes et Femmes": "HF",
    }
    df["sexe_code"] = df[sexe_col].map(sexe_map)
    df = df[df["sexe_code"].notna()]

    # 5) Create target variable name: {base}_{sexe}_{annee}
    df["var"] = (
        base + "_" + df["sexe_code"] + "_" + df[year_col].astype(int).astype(str)
    )

    # ==============================
    # 6) Aggregate Outre-mer (region_id 1..4 → 0)
    # ==============================
    mask_outre = df["region_id"].isin(list(outre_mer_regs))
    outre = df[mask_outre]

    if not outre.empty:
        g = outre.groupby([year_col, "sexe_code", "var"], as_index=False)[value_col]
        outre_agg = g.sum() if aggfunc == "sum" else g.mean()

        outre_agg["region_id"] = outre_mer_reg_value
        outre_agg["region"] = outre_mer_name

        outre_agg = outre_agg[
            ["region_id", "region", year_col, "sexe_code", "var", value_col]
        ]

        # remove original outre-mer rows, append aggregated
        df = df[~mask_outre]
        df = pd.concat([df, outre_agg], ignore_index=True)

    # ==============================
    # 7) Pivot to wide
    # ==============================
    out = (
        df.pivot_table(
            index=["region_id", "region"],
            columns="var",
            values=value_col,
            aggfunc="mean",  # safety
        )
        .reset_index()
    )

    out.columns.name = None
    return out



def compute_richesse(df1_fixed, df2_fixed, years=range(2014, 2024), keys=("region_id", "region")):
    m = df1_fixed.merge(df2_fixed, on=list(keys), how="inner", suffixes=("_df1", "_df2"))

    # start from all columns (so you keep montants_* and foyers_*)
    out = m.copy()

    for y in years:
        col_m = f"montants_{y}"
        col_f = f"foyers_{y}"
        col_r = f"richesse_{y}"

        # handle possible suffixes only if they exist
        m_col = col_m if col_m in out.columns else col_m + "_df2"
        f_col = col_f if col_f in out.columns else col_f + "_df1"

        if m_col not in out.columns or f_col not in out.columns:
            continue

        num = pd.to_numeric(out[m_col], errors="coerce")
        denom = pd.to_numeric(out[f_col], errors="coerce").mask(lambda x: x == 0)

        out[col_r] = num / denom

    return out



def clean_BAAC_region(BAAC_table_norm, add_region_name=True):
    df = BAAC_table_norm.copy()

    # ---------------------------
    # 1) dep (float) -> dep_int -> dep_code
    # ---------------------------
    dep_int = pd.to_numeric(df["dep"], errors="coerce").astype("Int64")
    df = df[dep_int.notna()].copy()
    df["dep_int"] = dep_int[dep_int.notna()].astype(int)

    # 2-digit department code (01..95) for mainland
    df["dep_code"] = df["dep_int"].astype(str).str.zfill(2)

    # ---------------------------
    # 2) dep -> region_id (INSEE)
    # ---------------------------
    # DOM: 971..976 etc.
    df["region_id"] = np.nan
    df.loc[df["dep_int"] >= 970, "region_id"] = 1      # Outre-mer
    df.loc[df["dep_int"] == 20,  "region_id"] = 94     # Corse (when coded as 20)

    # mainland mapping using dep_code
    dep = df["dep_code"]

    df.loc[dep.isin(["75","77","78","91","92","93","94","95"]), "region_id"] = 11
    df.loc[dep.isin(["18","28","36","37","41","45"]), "region_id"] = 24
    df.loc[dep.isin(["21","25","39","58","70","71","89","90"]), "region_id"] = 27
    df.loc[dep.isin(["14","27","50","61","76"]), "region_id"] = 28
    df.loc[dep.isin(["02","59","60","62","80"]), "region_id"] = 32
    df.loc[dep.isin(["08","10","51","52","54","55","57","67","68","88"]), "region_id"] = 44
    df.loc[dep.isin(["44","49","53","72","85"]), "region_id"] = 52
    df.loc[dep.isin(["22","29","35","56"]), "region_id"] = 53
    df.loc[dep.isin(["16","17","19","23","24","33","40","47","64","79","86","87"]), "region_id"] = 75
    df.loc[dep.isin(["09","11","12","30","31","32","34","46","48","65","66","81","82"]), "region_id"] = 76
    df.loc[dep.isin(["01","03","07","15","26","38","42","43","63","69","73","74"]), "region_id"] = 84
    df.loc[dep.isin(["04","05","06","13","83","84"]), "region_id"] = 93

    df = df[df["region_id"].notna()].copy()
    df["region_id"] = df["region_id"].astype(int)

    # ---------------------------
    # 3) sexe -> S (H/F), aggregate only needed metrics
    # ---------------------------
    sexe_map = {1: "H", 2: "F"}
    df["S"] = df["sexe"].map(sexe_map)
    df = df[df["S"].notna()].copy()

    df["grav"] = pd.to_numeric(df["grav"], errors="coerce")
    df["an_nais"] = pd.to_numeric(df["an_nais"], errors="coerce")

    panel = (
        df.groupby(["region_id", "year", "S"], as_index=False)
          .agg(
              sum_acc=("Num_Acc", "count"),
              avg_grav=("grav", "mean"),
              avg_an_nais=("an_nais", "mean"),
          )
    )

    if add_region_name:
        id_to_region = {
            11: "Île-de-France",
            24: "Centre-Val de Loire",
            27: "Bourgogne-Franche-Comté",
            28: "Normandie",
            32: "Hauts-de-France",
            44: "Grand Est",
            52: "Pays de la Loire",
            53: "Bretagne",
            75: "Nouvelle-Aquitaine",
            76: "Occitanie",
            84: "Auvergne-Rhône-Alpes",
            93: "Provence-Alpes-Côte d'Azur",
            94: "Corse",
            1:  "Outre mer",
        }
        panel["region"] = panel["region_id"].map(id_to_region)

    # keep only what you asked (no extra vars)
    return panel[["region_id", "region", "year", "S", "sum_acc", "avg_grav", "avg_an_nais"]]



def reshape_BAAC_region(panel_df, years=range(2020, 2025), sexes=("H", "F")):
    df = panel_df.copy()

    # Keep only requested years and sexes
    df = df[df["year"].isin(years) & df["S"].isin(sexes)].copy()

    # Prepare long → wide
    df_long = df.melt(
        id_vars=["region_id", "region", "year", "S"],
        value_vars=["sum_acc", "avg_grav", "avg_an_nais"],
        var_name="var",
        value_name="value"
    )

    # Build target column name: var_{year}_{sexe}
    df_long["col"] = (
        df_long["var"]
        + "_"
        + df_long["year"].astype(int).astype(str)
        + "_"
        + df_long["S"]
    )

    # Pivot to wide
    out = (
        df_long
        .pivot_table(
            index=["region_id", "region"],
            columns="col",
            values="value",
            aggfunc="first"
        )
        .reset_index()
    )

    out.columns.name = None
    return out

REGION_REF = {
        1:  "OUTRE-MER",
        11: "ÎLE-DE-FRANCE",
        24: "CENTRE-VAL DE LOIRE",
        27: "BOURGOGNE-FRANCHE-COMTÉ",
        28: "NORMANDIE",
        32: "HAUTS DE FRANCE",
        44: "GRAND EST",
        52: "PAYS DE LA LOIRE",
        53: "BRETAGNE",
        75: "NOUVELLE AQUITAINE",
        76: "OCCITANIE",
        84: "AUVERGNE-RHÔNE-ALPES",
        93: "PROVENCE-ALPES-CÔTE D'AZUR",
        94: "CORSE",
    }



def fix_region(
    df,
    REGION_REF = REGION_REF,
    region_id_col="region_id",
    region_col="region",
    strict_ids=True
):
    """
    Ensures (region_id, region) matches REGION_REF exactly.
    - REGION_REF: dict {int region_id: str canonical_region_name}
    - Overwrites `region` using region_id when possible (source of truth).
    - Can also infer/repair region_id from region string when possible.
    """
    df = df.copy()

    # 1) Required columns
    if region_id_col not in df.columns or region_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{region_id_col}' and '{region_col}'")

    # 2) Normalize region_id to int where possible
    df[region_id_col] = pd.to_numeric(df[region_id_col], errors="coerce")
    if df[region_id_col].isna().any():
        bad = df.loc[df[region_id_col].isna(), region_col].head(10).tolist()
        raise ValueError(f"Some {region_id_col} could not be converted to numeric. Examples: {bad}")
    df[region_id_col] = df[region_id_col].astype(int)

    # 3) Normalize region strings
    df[region_col] = df[region_col].astype(str).str.strip().str.upper()

    # small alias normalization (you can extend if needed)
    aliases = {
        "OUTRE MER": "OUTRE-MER",
        "OUTREMER": "OUTRE-MER",
        "DOM": "OUTRE-MER",
        "DOM-TOM": "OUTRE-MER",
        "DOM TOM": "OUTRE-MER",
        "ILE-DE-FRANCE": "ÎLE-DE-FRANCE",
        "ILE DE FRANCE": "ÎLE-DE-FRANCE",
        "CENTRE VAL DE LOIRE": "CENTRE-VAL DE LOIRE",
        "BOURGOGNE ET FRANCHE-COMTÉ": "BOURGOGNE-FRANCHE-COMTÉ",
        "BOURGOGNE ET FRANCHE COMTÉ": "BOURGOGNE-FRANCHE-COMTÉ",
        "BOURGOGNE ET FRANCHE COMTE": "BOURGOGNE-FRANCHE-COMTÉ",
        "BOURGOGNE FRANCHE COMTE": "BOURGOGNE-FRANCHE-COMTÉ",
        "AUVERGNE ET RHÔNE-ALPES": "AUVERGNE-RHÔNE-ALPES",
        "AUVERGNE ET RHONE-ALPES": "AUVERGNE-RHÔNE-ALPES",
        "AUVERGNE RHONE ALPES": "AUVERGNE-RHÔNE-ALPES",
        "NOUVELLE AQUITAINE": "NOUVELLE AQUITAINE",
        "NOUVELLE-AQUITAINE": "NOUVELLE AQUITAINE",
        "PROVENCE-ALPES-COTE D'AZUR": "PROVENCE-ALPES-CÔTE D'AZUR",
        "PROVENCE ALPES COTE D AZUR": "PROVENCE-ALPES-CÔTE D'AZUR",
        "HAUTS-DE-FRANCE": "HAUTS DE FRANCE",
    }
    df[region_col] = df[region_col].replace(aliases)

    # 4) Build reverse map from canonical region names
    id_to_region = {int(k): str(v).upper() for k, v in REGION_REF.items()}
    region_to_id = {v: k for k, v in id_to_region.items()}

    # 5) Fix region from region_id whenever possible (source of truth)
    mask_known_id = df[region_id_col].isin(id_to_region.keys())
    df.loc[mask_known_id, region_col] = df.loc[mask_known_id, region_id_col].map(id_to_region)

    # 6) Fix region_id from region whenever possible (helps if ids are wrong but names ok)
    mask_known_region = df[region_col].isin(region_to_id.keys())
    df.loc[mask_known_region, region_id_col] = df.loc[mask_known_region, region_col].map(region_to_id)

    # 7) Validation
    unknown_ids = sorted(set(df[region_id_col]) - set(id_to_region.keys()))
    if strict_ids and unknown_ids:
        raise ValueError(f"Unknown {region_id_col}(s) found: {unknown_ids}")

    bad = df[df[region_id_col].map(id_to_region) != df[region_col]]
    if not bad.empty:
        raise ValueError(
            "Inconsistent region_id / region after correction:\n"
            + bad[[region_id_col, region_col]].drop_duplicates().to_string(index=False)
        )

    return df
