import requests
import pandas as pd
import numpy as np
import re
import io
from functools import reduce




def get_resources(api_url: str) -> pd.DataFrame:
    """
    BAAC dataset endpoint -> resources DataFrame (one row per resource).
    """
    payload = requests.get(api_url, timeout=30).json()
    return pd.json_normalize(payload["resources"])



def select_baac_tables(BAAC_resources: pd.DataFrame) -> pd.DataFrame:
    df = BAAC_resources.copy()

    pattern = re.compile(
        r"/(carcteristiques|caract|caracteristiques|lieux|vehicules|usagers)-(202[0-4])\.csv$",
        re.IGNORECASE
    )

    m = df["url"].astype(str).str.extract(pattern)
    out = df[m[0].notna()].copy()

    out["table"] = m.loc[m[0].notna(), 0].str.lower()
    out["year"] = m.loc[m[0].notna(), 1].astype(int)

    out = out[(out["year"] >= 2020) & (out["year"] <= 2024)]

    # normalize table names
    out["table"] = out["table"].replace(
        {"carcteristiques": "caracteristiques", "caract": "caracteristiques"}
    )

    return out[["description", "url", "table", "year"]].reset_index(drop=True)



def build_baac_dataframe(selected_BAAC_table: pd.DataFrame) -> pd.DataFrame:
    t = selected_BAAC_table.copy()

    def _load(u):
        # read with low_memory=False to avoid mixed-type dtype warnings
        df = pd.read_csv(u, sep=";", low_memory=False)
        df.columns = df.columns.astype(str).str.strip()
        if "Accident_Id" in df.columns:
            df = df.rename(columns={"Accident_Id": "Num_Acc"})
        return df

    t["df"] = t["url"].apply(_load)

    yearly = []
    for year, g in t.groupby("year"):
        dfs = g["df"].tolist()
        merged = reduce(lambda left, right: left.merge(right, on="Num_Acc", how="outer"), dfs)
        merged["year"] = year

        def coalesce_and_clean(df, base):
            # find columns that contain base (like 'long', 'long_x', 'long_y', ...)
            cand = [c for c in df.columns if re.search(rf'\b{base}\b', c)]
            if not cand:
                return df

            # Clean candidate columns into numeric series first
            cleaned = {}
            for c in cand:
                s = df[c].astype(str).str.strip().str.replace(',', '.', regex=False)
                s = s.replace(r'^[^\d\-\.+].*$', np.nan, regex=True)
                cleaned[c] = pd.to_numeric(s, errors='coerce')

            # Heuristic per-row selector: prefer values in plausible ranges for lat/long,
            # otherwise choose the candidate with largest absolute value (to avoid tiny scaled vals).
            def pick_value(row_idx):
                vals = {c: cleaned[c].iat[row_idx] for c in cand}
                # remove nans
                vals = {k: v for k, v in vals.items() if not pd.isna(v)}
                if not vals:
                    return np.nan
                # plausible ranges
                if base == "lat":
                    good = {k: v for k, v in vals.items() if 41 <= v <= 51}
                else:  # long
                    good = {k: v for k, v in vals.items() if -6 <= v <= 10}
                if good:
                    # if several plausible, pick the one with median magnitude (stable choice)
                    return list(good.values())[0]
                # no plausible found: pick candidate with largest absolute magnitude
                return max(vals.values(), key=lambda x: abs(x))

            picked = [pick_value(i) for i in range(len(df))]
            df[base] = pd.Series(picked, index=df.index, dtype="float64")

            # drop redundant candidate columns
            to_drop = [c for c in cand if c != base]
            df = df.drop(columns=to_drop)
            return df

        merged = coalesce_and_clean(merged, "lat")
        merged = coalesce_and_clean(merged, "long")

        yearly.append(merged)

    return pd.concat(yearly, ignore_index=True)



def add_data(df) : 

    np.random.seed(42)  # reproducibility

    N = len(df)

    #CSP
    csp_categories = [
        "Cadres / professions supérieures",
        "Professions intermédiaires",
        "Employés",
        "Ouvriers",
        "Agriculteurs",
        "Indépendants / artisans",
        "Chômeurs",
        "Retraités"
    ]

    raw_probabilities = [
        0.075,  # cadres (lowest risk)
        0.11,
        0.13,
        0.15,
        0.154,
        0.12,
        0.134,
        0.167   # retraités
    ]

    csp_probabilities = raw_probabilities / np.array(raw_probabilities).sum()

    df["csp_conducteur"] = np.random.choice(
        csp_categories,
        size=N,
        p=csp_probabilities
    )

    return df
