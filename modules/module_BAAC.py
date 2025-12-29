import requests
import pandas as pd
import numpy as np
import re
import io



def get_resources(api_url: str) -> pd.DataFrame:
    """
    BAAC dataset endpoint -> resources DataFrame (one row per resource).
    """
    payload = requests.get(api_url, timeout=30).json()
    return pd.json_normalize(payload["resources"])



def select_baac_tables(BAAC_resources: pd.DataFrame) -> pd.DataFrame: 
    """
    From BAAC_resources[['description','url']], keep only yearly BAAC tables whose URL ends with:
      - carcteristiques-YYYY.csv
      - caract-YYYY.csv
      - usagers_YYYY.csv
      - vehicules_YYYY.csv
      - lieux_YYYY.csv
      - caracteristiques_YYYY.csv

    Returns a DataFrame with columns: description, url, table, year
    """
    df = BAAC_resources[["description", "url"]].copy()
    df["url"] = df["url"].astype(str)

    pattern = re.compile(
        r"(?:caract|caracteristiques|carcteristiques|lieux|vehicules|usagers)[-_]\d{4}\.csv$",
        flags=re.IGNORECASE
    )

    out = df[df["url"].str.contains(pattern, na=False)].copy()

    # Extract table name + year
    def parse(url: str):
        m = re.search(r"(carcteristiques|caract|caracteristiques|usagers|vehicules|lieux)[-_](\d{4})\.csv$", url, re.I)
        if not m:
            return pd.Series([None, None])
        table, year = m.group(1).lower(), int(m.group(2))
        # normalize table name
        if table in ("carcteristiques", "caracteristiques", "caract"):
            table = "caracteristiques"
        return pd.Series([table, year])

    out[["table", "year"]] = out["url"].apply(parse)

    
    return out.reset_index(drop=True)



def _read_baac_csv(url: str) -> pd.DataFrame:
    # encoding fallback
    try:
        df = pd.read_csv(url, sep=None, engine="python", dtype=str, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(url, sep=None, engine="python", dtype=str, encoding="latin1")

    # packed 1-col repair
    if df.shape[1] == 1 and df.iloc[:, 0].astype(str).str.contains(",").mean() > 0.3:
        text = df.columns[0] + "\n" + "\n".join(df.iloc[:, 0].astype(str).tolist())
        df = pd.read_csv(io.StringIO(text), sep=",", engine="python", dtype=str)

    # normalize column names
    df.columns = (
        pd.Index(df.columns)
        .astype(str)
        .str.strip()
        .str.replace("\ufeff", "", regex=False)  # BOM sometimes
    )

    # force rename common variants -> Num_Acc
    for c in df.columns:
        if c.replace(" ", "").lower() in {"num_acc", "numacc", "num_accident", "numaccident"}:
            df = df.rename(columns={c: "Num_Acc"})
            break

    return df



def build_baac_dataframe(selected_BAAC_table: pd.DataFrame) -> pd.DataFrame:
    t = selected_BAAC_table.copy()

    def _load(u):
        df = pd.read_csv(u, sep=";")
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
        yearly.append(merged)
        print(f'{year} completed')
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
