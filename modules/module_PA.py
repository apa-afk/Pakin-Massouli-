import pandas as pd
import requests


def get_PA_tables(PA_resources: pd.DataFrame) -> pd.DataFrame:
    """
    Load the Parc Automobile dataset from its resources DataFrame.
    Assumes a single CSV resource.
    """
    url = PA_resources.loc[0, "url"]

    # 1) download raw text
    text = requests.get(url, timeout=30).text.strip()

    # 2) split lines
    lines = text.splitlines()
    if len(lines) < 2:
        raise ValueError("CSV content is invalid (less than 2 lines).")

    # 3) split header and values manually
    header = [h.strip('"') for h in lines[0].split(",")]
    rows = [[v.strip('"') for v in line.split(",")] for line in lines[1:]]

    # 4) build DataFrame
    df = pd.DataFrame(rows, columns=header)