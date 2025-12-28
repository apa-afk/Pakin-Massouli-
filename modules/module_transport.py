import pandas as pd
import requests
import ast


def get_resources(api_url: str) -> pd.DataFrame:
    """
    BAAC dataset endpoint -> resources DataFrame (one row per resource).
    """
    payload = requests.get(api_url, timeout=30).json()
    return pd.json_normalize(payload)



def extract_resources_urls(resources) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "url": f"https://transport.data.gouv.fr/resources/{d['id']}/download",
            "title": d.get("title")
        }
        for sub in resources          # liste
        for d in sub                  # dict
    ])
