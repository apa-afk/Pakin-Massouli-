import requests
import pandas as pd
import numpy as np


def get_raw_data_BAAC(api_url: str) -> pd.DataFrame:
    """
    BAAC dataset endpoint -> resources DataFrame (one row per resource).
    """
    payload = requests.get(api_url, timeout=30).json()
    return pd.json_normalize(payload["resources"])


def get_baac_tables(resources_df: pd.DataFrame, selected_table: list) -> pd.DataFrame:
    """
    From BAAC resources (title/format/url), pick the selected BAAC CSV tables, load them and fuse them
    to have one accident per line.
    """
    # 1) load selected csvs
    sel = resources_df.loc[resources_df["description"].isin(selected_table), ["description", "url"]]
    dfs = {d: pd.read_csv(u, sep=";", low_memory=False) for d, u in sel.to_records(index=False)}

    # 2) identify the 4 tables
    carac = next(v for k, v in dfs.items() if "caractéristiques" in k.lower())
    lieux = next(v for k, v in dfs.items() if "lieux" in k.lower())
    veh   = next(v for k, v in dfs.items() if "véhicules impliqués" in k.lower() or "vehicules impliques" in k.lower())
    usa   = next(v for k, v in dfs.items() if "usagers" in k.lower())

    # 3) merge accident-level tables
    for df in (carac, lieux, veh, usa):
        df["Num_Acc"] = df["Num_Acc"].astype(str)

    acc = carac.merge(lieux, on="Num_Acc", how="left")

    # 4) add simple enrichments (counts)
    acc["nb_vehicules"] = veh.groupby("Num_Acc").size().reindex(acc["Num_Acc"]).fillna(0).astype(int).values
    acc["nb_usagers"]   = usa.groupby("Num_Acc").size().reindex(acc["Num_Acc"]).fillna(0).astype(int).values

    # hour instead of hourmn
    acc['hour'] = acc['hrmn'].astype(str).str.zfill(4).str[:2].astype(int)


    return acc






def add_data(df) : 


    np.random.seed(42)  # reproducibility

    N = len(df)

    #AGE
    age_groups = [
        (18, 29, 0.35),
        (30, 44, 0.25),
        (45, 59, 0.20),
        (60, 74, 0.15),
        (75, 85, 0.05)
    ]

    ages = []
    for low, high, p in age_groups:
        count = int(p * N)
        ages.extend(np.random.randint(low, high + 1, count))

    ages = np.random.choice(ages, N, replace=True)
    df["age_conducteur"] = ages

    
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

    
    #FRENCH REGION
    regions = [
        "Île-de-France",
        "Nord",
        "Ouest",
        "Est",
        "Sud-Ouest",
        "Sud-Est",
        "Centre-Massif"
    ]

    # Approximate population + traffic exposure
    region_probabilities = [
        0.18,  # IDF
        0.17,
        0.15,
        0.14,
        0.13,
        0.15,
        0.08
    ]

    df["region"] = np.random.choice(
        regions,
        size=N,
        p=region_probabilities
    )

    return df