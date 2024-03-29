import os
import requests
import pandas as pd
from src.config import DATA_DIR, PROCESSED_DATA_PATH

def main():
    """
    Traversing through the data directory and subdirectories, to locate all
    png image files.
    Storing them as a csv file with relative path and labels.


    Parameters
    ----------
    None


    Returns
    -------
    None
    """
    imagePaths = []
    for dirname, _, filenames in os.walk(DATA_DIR):
        for filename in filenames:
            if (filename[-3:] == 'png'):
                imagePaths.append(os.path.join(dirname, filename))

    path = []
    label = []
    for image in imagePaths:
        path.append(image)
        label.append(image.split('/')[3])

    data = pd.DataFrame({'path':path,'label':label})
    data.to_csv(PROCESSED_DATA_PATH, index=False)

def live_data(key):
    """
    Requests latest covid-19 stats from rapidapi using requests.


    Parameters
    ----------
    key: rapidapi key


    Returns
    -------
    response: A dictionary containing data in json format.
    """
    # https://rapidapi.com/KishCom/api/covid-19-coronavirus-statistics/
    url = "https://covid-19-coronavirus-statistics.p.rapidapi.com/v1/stats"

    querystring = {"country":"Canada"}

    headers = {
        "X-RapidAPI-Key": key,
        "X-RapidAPI-Host": "covid-19-coronavirus-statistics.p.rapidapi.com"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)
    return response

if __name__ == '__main__':
    main()