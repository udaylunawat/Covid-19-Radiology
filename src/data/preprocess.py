import json
import numpy as np
import pandas as pd

def covid_stats(res):
    data = json.loads(res.text) 

    country_wise = pd.DataFrame(data['countries_stat'])
    country_wise = country_wise.replace('', np.nan).fillna(0)
    country_wise = country_wise[['country_name','cases','deaths','deaths_per_1m_population']]
    country_wise['cases'] = country_wise['cases'].str.replace(',', '').astype(int)
    country_wise['deaths'] = country_wise['deaths'].str.replace(',', '').astype(int)
    country_wise['deaths_per_1m_population'] = country_wise['deaths_per_1m_population'].str.replace(',', '').astype(float)
    country_wise = country_wise.astype({'cases': 'int', 'deaths': 'int', 'deaths_per_1m_population':'float64'})
    update = data['statistic_taken_at']
    return country_wise, update