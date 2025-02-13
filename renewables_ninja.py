from datetime import datetime, timedelta
import requests
import json
import io
import pandas as pd

n_days = 14 # MUST BE GREATER THAN 1
n_hours = 24 * n_days
startdate = "2023-01-01"

# Convert string to datetime object, add n_days and convert it back
startdate_dt = datetime.strptime(startdate, "%Y-%m-%d")
enddate_dt = startdate_dt + timedelta(days=n_days-1)
enddate = enddate_dt.strftime("%Y-%m-%d")


token = 'd0c7b389b3a5523e6d4c23c64776e0539ad5e543'
url_base = 'https://www.renewables.ninja/api/'
url = url_base + 'data/pv'

session = requests.session()
# Send token header with each request
session.headers = {'Authorization': 'Token ' + token}

args = {
    "lat": 48.2082,   # Example: Vienna, Austria
    "lon": 16.3738,
    "date_from": startdate,
    "date_to": enddate,
    "dataset": "merra2",  # Options: "merra2" or "era5"
    "capacity": 1.0,
    "system_loss": 0.0,
    "tracking": 0,  
    "tilt": 35,
    "azim": 180,
    "format": "json"
}
# Request data from https://www.renewables.ninja/api/
request = session.get(url, params=args)

# Parse JSON to get a pandas.DataFrame of data and dict of metadata
parsed_response = json.loads(request.text)

json_data = json.dumps(parsed_response["data"])  # Convert dictionary to JSON string
data = pd.read_json(io.StringIO(json_data), orient="index")
metadata = parsed_response['metadata']
# Convert JSON to Pandas DataFrame
pv_profile = pd.DataFrame(data)
pv_profile.index = pd.to_datetime(pv_profile.index)

# Resample to hourly values
pv_profile_hourly = pv_profile.resample("h").mean()

# Save to CSV for later use
pv_profile_hourly.to_csv("pv_profile_hourly.csv")

# Load saved Renewables.ninja PV profile
pv_profile_RenewablesNinja = pd.read_csv("pv_profile_hourly.csv", index_col=0, parse_dates=True)

pv_profile = pv_profile_RenewablesNinja
# Convert the pv profile to a list for PyPSA 
if isinstance(pv_profile, pd.Series):
    pv_profile_list = pv_profile.tolist()
else:
    pv_profile_list = pv_profile.iloc[:, 0].tolist()
