import requests
import pandas as pd




def get_raw_data(api_url):
    """
    Calls a public API to retrieve data in DataFrame format.
    Parameters:
    - api_url (str): The URL of the API.
    Returns:
    DataFrame: The data in DataFrame format.
    """
    response = requests.get(api_url)
    data = response.json()
    return pd.DataFrame(data)

