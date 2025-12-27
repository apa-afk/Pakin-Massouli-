import requests
import pandas as pd



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
    acc['hour'] = acc['hrmn'].str.split(':').str[0].astype(int)


    return acc