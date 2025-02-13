def PV_estimater(coordinates, dates, PV_peak = 1, loss = 0.1, tracking = 0, tilt = 0, azim = 180, debug = False):
    """ Estimates the PV capacity from startdate to enddate for a period, if in future it gets data from year 2024 only gets the dates which are requested 
        eg. when asked from 2023-01-01 04:00 - 2023-01-01 06:00 it displays only in that range
    Input:
        - coordinates:  Coordinates in format [latitude, longitude]
        - dates:        Vector with [Startdate, Enddate], either in UNIX time or in Format like "2023-01-01 00:00" or "2023-01-01"
        - PV_peak:      installed peak power of PV
        - loss:         Combined losses of all components (Panel, Converter etc.)
        - tracking:     either 0: no tracking; 1: azimuth tracking; 2: tilt & azimuth tracking
        - tilt:         angle of horizontal tilt; 0 = facing upwards; 90 = facing sideways
        - azim:         azimuth angle; 180 = poleward facing (in northern hemisphere = south)
        - debug:        Prints debugging info, like Startdate & Enddate; Answer gained from website
    Output:
        - Date:         Vector with dates
        - PV_power:     Vector with corresponding PV power to date

    ToDo: 
    Take mean of the days from last n years to increase accuracy!
        """
    from datetime import datetime
    import requests
    if debug:
        print("DEBUGGING MODE ENABLED")

    #Datetime handling, either UNIX to datetime or string to datetime
    if type(dates[0]) == type(0.2):
        startdate = datetime.fromtimestamp(dates[0])
        enddate = datetime.fromtimestamp(dates[1])
    else:
        try:
            startdate = datetime.strptime(dates[0],  "%Y-%m-%d %H:%M")
            enddate = datetime.strptime(dates[1],  "%Y-%m-%d %H:%M")
        except:
            startdate = datetime.strptime(dates[0],  "%Y-%m-%d")
            enddate = datetime.strptime(dates[1],  "%Y-%m-%d")

    if debug:
        print("Startdate & Enddate: ")
        print(startdate, enddate)

    # duration = enddate - startdate
    # n_days = duration / (24*3600)

    #check if too recent --> change year where data available
    if startdate.year > 2024:
        startdate = startdate.replace(year=2024)
    
    if enddate.year > 2024:
        enddate = enddate.replace(year=2024)

    #token = 'd0c7b389b3a5523e6d4c23c64776e0539ad5e543'  #Sebastian Token
    token = 'c380db12bbe749a2fed33b52f1c0297006fbcef8'  #Clemens Token
    url_base = 'https://www.renewables.ninja/api/'
    url = url_base + 'data/pv'

    #Open session & make request with arguments
    session = requests.session()
    session.headers = {'Authorization': 'Token ' + token}
    args = {
        "lat": coordinates[0],   # Example: Vienna, Austria
        "lon": coordinates[0],
        "date_from": startdate.strftime("%Y-%m-%d"),
        "date_to": enddate.strftime("%Y-%m-%d"),
        "dataset": "merra2",  # Options: "merra2" or "era5"
        "capacity": PV_peak,
        "system_loss": loss,
        "tracking": tracking,  
        "tilt": tilt,
        "azim": azim,
        "format": "csv"
    }
    request = session.get(url, params=args)
    if debug:
        print("DEBUGGING! RESPONSE: ")
        print(request.text)

#When length of the request is below certain character count --> Error because over hour/ second limit etc.
    if len(request.text) <= 150:
        print("Potential Error encountered!")
        print(request.text)
    ###### DATA MANIPULATION #####
    # Split lines and remove the header
    lines = request.text.split("\n")[4:]

    # Extract time and electricity values
    time_list = []
    electricity_list = []
    for line in lines:
        if ',' in line: #this needs to be included to get rid of last empty line
            time, electricity = line.split(",")
            time_list.append(time)
            electricity_list.append(float(electricity))  # Convert to float

    time_dt_list = [datetime.strptime(t, "%Y-%m-%d %H:%M") for t in time_list]


    # Filter both lists based on startdate and enddate
    filtered_time_list = []
    filtered_electricity_list = []

    for i in range(len(time_dt_list)):
        if startdate <= time_dt_list[i] <= enddate:
            filtered_time_list.append(time_list[i])  # Keep original string format
            filtered_electricity_list.append(electricity_list[i])
    return filtered_time_list, filtered_electricity_list

if __name__ == "__main__":
    import time
    time1, elec1 = PV_estimater([39.0340, 12.2852], [time.time(), time.time()+7200], 60, debug = True)
    
    time.sleep(1)
    time2, elec2 = PV_estimater([39.0340, 12.2852], ["2025-02-12", "2025-02-13"], 60)
    time.sleep(1)
    time3, elec3 = PV_estimater([39.0340, 12.2852], ["2025-02-12 08:00", "2025-02-12 16:00"], 60, debug=False)


    print(time1, elec1)
    print(time2, elec2)
    print(time3, elec3)