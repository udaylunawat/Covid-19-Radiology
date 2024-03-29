import json
import pytz
import numpy as np
import pandas as pd
from datetime import datetime, timezone

def covid_stats(res, timezone):
    '''
    Preprocesses data collected from rapid api and returns processed data and ,date and time of data generation.

    Parameters
    ----------
    res : response value returned by live_data function in make_dataset.py


    Returns
    -------
    country_wise : Latest Country wise covid-19 data with country name, cases, deaths and deaths per million population.
    update: date and time (GMT +0) when data was generated on rapidapi.
    '''
    data = json.loads(res.text)

    country_wise = pd.DataFrame(data['data']['covid19Stats'])
    country_wise = country_wise.replace('', np.nan).fillna(0)
    country_wise = country_wise[['country','confirmed','deaths']]
    country_wise['country'] = country_wise['country'].str.replace(',', '').astype(str)
    # country_wise['deaths'] = country_wise['deaths'].str.replace(',', '').astype(int)
    # country_wise['confirmed'] = country_wise['confirmed'].str.replace(',', '').astype(int)
    country_wise = country_wise.astype({'confirmed': 'int', 'deaths': 'int', 'country':'str'})
    updated_at = data['lastUpdate'][0]
    updated_datetime = change_timezone(updated_at, timezone)
    return country_wise, updated_datetime


def utc_to_local(utc_dt, tz):
    '''
    Updated passed datetime to a passed timezone

    Parameters
    ----------
    utc_dt : datetime formatted date & time
    tz : timezone in pytz.timezone format

    Returns
    -------
    string : date and time updated to the passed timezone.
    '''

    return utc_dt.replace(tzinfo=timezone.utc).astimezone(tz=tz)

def change_timezone(updated_at, timezone = 'Asia/Kolkata'):
    '''
    Accepts datetime as string type in format '%Y-%m-%d %H:%M:%S' and updates it as passed timezone.
    Default timezone is Indian.

    Parameters
    ----------
    utc_dt : datetime formatted date & time
    tz : timezone in pytz.timezone format

    Returns
    -------
    updated_datetime : date and time updated to the passed timezone.
    '''

    dt_format = datetime.strptime(updated_at, '%Y-%m-%d %H:%M:%S')
    local_tz = pytz.timezone(timezone)
    fmt = '%m-%d-%Y %H:%M:%S %Z'

    updated_datetime = utc_to_local(dt_format, local_tz).strftime(fmt)
    return updated_datetime