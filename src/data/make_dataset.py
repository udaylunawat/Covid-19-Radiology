import os
import requests
import pandas as pd
from src.config import rapid_api_key, DATA_DIR, PROCESSED_DATA_PATH

def main():
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
    # https://rapidapi.com/astsiatsko/api/coronavirus-monitor
    url = "https://coronavirus-monitor.p.rapidapi.com/coronavirus/cases_by_country.php"

    headers = {
        'x-rapidapi-host': "coronavirus-monitor.p.rapidapi.com",
        'x-rapidapi-key': rapid_api_key
        }

    response = requests.request("GET", url, headers=headers)
    return response

if __name__ == '__main__':
    main()