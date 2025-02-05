# CHANGE LOG:
# V1.0
# First stable prototype
# Implemented single power source (PV)
# Implemented single consumer (constant load)
# Implemented variable power generation (sinusoidal curve)
# Implemented battery storage
# Configured assets for optimization algorithm
# Implemented visualization of results
# Added new variable for curtailment control
# Added custom constraint to linopy for control of maximum curtailment of power generation

# V2.0
# Integrated LoadProfileGenerator (PyLPG)
# Implemented generation of multiple pre-defined households

# V2.1
# Integrated renewables.ninja data
# Adjusted timeframing of simulation to enable renewables.ninja data

# V2.2
# Integrated RAMP load curve generation
# Synchronized datetimes for RAMP and PyPSA
# Combined LPG and RAMP load curves into a total load on the power grid

# ---------------------------------TODO:------------------
# Integrate "Standardlastprofile"
# Implement addition of new consumers
# Implement a parser to control the simulation 


from ramp.post_process import post_process as pp
from ramp.core.core import UseCase
from ramp.core.core import User

import pypsa
from pylpg import lpg_execution, lpgdata

import requests
import json
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

filename_LPG = "community_load_profile.csv"
n_days = 21 # MUST BE GREATER THAN 1
n_hours = 24 * n_days
startdate = "2023-01-01"

# --------------------------
# Integration of RAMP for load curve generation
# --------------------------

# RAMP does not deal with units of measures, you should check the consistency of the unit of measures throughout your model
User_list = []

# Create new user classes
Hospital = User("hospital", 1)
User_list.append(Hospital)

School = User("school", 1)
User_list.append(School)

Public_lighting = User("public lighting", 1)
User_list.append(Public_lighting)

Church = User("church", 3)
User_list.append(Church)

# Create new appliances
# Church
Ch_indoor_bulb = Church.add_appliance(
    number=10, # how many of this appliance each user has in this user category
    power=26, # the power [Watt] of each single appliance. 
    num_windows=1, # how many usage time windows throughout the day?
    func_time=210, # the total usage time of appliances [min]
    time_fraction_random_variability=0.2, # randomizes the total time the appliance is on (between 0 and 1)
    func_cycle=60, # the minimum usage time after a switch on event [min]
    fixed="yes", # This means all the 'n' appliances of this kind are always switched-on together
    flat="yes", # This means the appliance is not subject to random variability in terms of total usage time
    name="indoor_bulb",
)
Ch_indoor_bulb.windows(window_1=[1200, 1440], window_2=[0, 0], random_var_w=0.1) # from 20:00 to 24:00

Ch_outdoor_bulb = Church.add_appliance(
    7, 26, 1, 150, 0.2, 60, "yes", flat="yes", name="outdoor_bulb"
)
Ch_outdoor_bulb.windows([1200, 1440], [0, 0], 0.1) 

Ch_speaker = Church.add_appliance(1, 100, 1, 150, 0.2, 60, name="speaker")
Ch_speaker.windows([1200, 1350], [0, 0], 0.1)

# Public lighting
Pub_lights = Public_lighting.add_appliance(
    12, 40, 2, 310, 0.1, 300, "yes", flat="yes", name="lights"
)
Pub_lights.windows([0, 336], [1110, 1440], 0.2)

Pub_lights_2 = Public_lighting.add_appliance(
    25, 150, 2, 310, 0.1, 300, "yes", flat="yes", name="lights2"
)
Pub_lights_2.windows([0, 336], [1110, 1440], 0.2)

# Hospital
Ho_indoor_bulb = Hospital.add_appliance(12, 7, 2, 690, 0.2, 10, name="indoor_bulb")
Ho_indoor_bulb.windows([480, 720], [870, 1440], 0.35)

Ho_outdoor_bulb = Hospital.add_appliance(1, 13, 2, 690, 0.2, 10, name="outdoor_bulb")
Ho_outdoor_bulb.windows([0, 330], [1050, 1440], 0.35)

Ho_Phone_charger = Hospital.add_appliance(8, 2, 2, 300, 0.2, 5, name="phone_charger")
Ho_Phone_charger.windows([480, 720], [900, 1440], 0.35)

# specific_cycle_1 is used for assigining the first specific duty cycle for the appliace (maximum of three cycles can be assigned)
# Parameters:
    # p_11 : float, optional
    # Power rating for first part of first duty cycle. Only necessary if fixed_cycle is set to 1 or greater, by default 0
    # t_11 : int[0,1440], optional
    # Duration (minutes) of first part of first duty cycle. Only necessary if fixed_cycle is set to 1 or greater, by default 0
    # p_12 : int, float, optional
    # Power rating for second part of first duty cycle. Only necessary if fixed_cycle is set to 1 or greater, by default 0
    # t_12 : int[0,1440], optional
    # Duration (minutes) of second part of first duty cycle. Only necessary if fixed_cycle is set to 1 or greater, by default 0
    # r_c1 : Percentage [0,1], optional
    # randomization of the duty cycle parts duration. There will be a uniform random variation around t_i1 and t_i2. If this parameter is set to 0.1, then t_i1 and t_i2 will be randomly reassigned between 90% and 110% of their initial value; 0 means no randomisation, by default 0
    # w11 : Iterable, optional
    # Window time range for the first part of first duty cycle number (not neccessarily linked to the overall time window), by default None
    # cw12 : Iterable, optional
    # Window time range for the first part of first duty cycle number (not neccessarily linked to the overall time window), by default None, by default None     

Ho_Fridge = Hospital.add_appliance(1, 150, 1, 1440, 0, 30, "yes", 3, name="fridge")
Ho_Fridge.windows([0, 1440], [0, 0])
Ho_Fridge.specific_cycle_1(p_11= 150,t_11= 20, p_12=5, t_12=10)
Ho_Fridge.specific_cycle_2(150, 15, 5, 15)
Ho_Fridge.specific_cycle_3(150, 10, 5, 20)
Ho_Fridge.cycle_behaviour(
    cw11=[580, 1200], cw12=[0, 0], cw21=[420, 579], cw22=[0, 0], cw31=[0, 419], cw32=[1201, 1440]
)
# different time windows can be associated with different specific duty cycles
# Parameters
# cw11 : Iterable, optional
# Window time range for the first part of first duty cycle number, by default np.array([0,0])
# cw12 : Iterable, optional
# Window time range for the second part of first duty cycle number, by default np.array([0,0])
# cw21 : Iterable, optional
# Window time range for the first part of second duty cycle number, by default np.array([0,0])
# cw22 : Iterable, optional
# Window time range for the second part of second duty cycle number, by default np.array([0,0])
# cw31 : Iterable, optional
# Window time range for the first part of third duty cycle number, by default np.array([0,0])
# cw32 : Iterable, optional
# Window time range for the second part of third duty cycle number, by default np.array([0,0])

Ho_Fridge2 = Hospital.add_appliance(1, 150, 1, 1440, 0, 30, "yes", 3, name="fridge2")
Ho_Fridge2.windows([0, 1440], [0, 0])
Ho_Fridge2.specific_cycle_1(150, 20, 5, 10)
Ho_Fridge2.specific_cycle_2(150, 15, 5, 15)
Ho_Fridge2.specific_cycle_3(150, 10, 5, 20)
Ho_Fridge2.cycle_behaviour(
    [580, 1200], [0, 0], [420, 579], [0, 0], [0, 419], [1201, 1440]
)

Ho_Fridge3 = Hospital.add_appliance(1, 150, 1, 1440, 0.1, 30, "yes", 3, name="fridge3")
Ho_Fridge3.windows([0, 1440], [0, 0])
Ho_Fridge3.specific_cycle_1(150, 20, 5, 10)
Ho_Fridge3.specific_cycle_2(150, 15, 5, 15)
Ho_Fridge3.specific_cycle_3(150, 10, 5, 20)
Ho_Fridge3.cycle_behaviour(
    [580, 1200], [0, 0], [420, 579], [0, 0], [0, 419], [1201, 1440]
)

Ho_PC = Hospital.add_appliance(2, 50, 2, 300, 0.1, 10, name="PC")
Ho_PC.windows([480, 720], [1050, 1440], 0.35)

Ho_Mixer = Hospital.add_appliance(
    1, 50, 2, 60, 0.1, 1, occasional_use=0.33, name="mixer"
)
Ho_Mixer.windows([480, 720], [1050, 1440], 0.35)

# School
S_indoor_bulb = School.add_appliance(8, 7, 1, 60, 0.2, 10, name="indoor_bulb")
S_indoor_bulb.windows([1020, 1080], [0, 0], 0.35)

S_outdoor_bulb = School.add_appliance(6, 13, 1, 60, 0.2, 10, name="outdoor_bulb")
S_outdoor_bulb.windows([1020, 1080], [0, 0], 0.35)

S_Phone_charger = School.add_appliance(5, 2, 2, 180, 0.2, 5, name="phone_charger")
S_Phone_charger.windows([510, 750], [810, 1080], 0.35)

S_PC = School.add_appliance(18, 50, 2, 210, 0.1, 10, name="PC")
S_PC.windows([510, 750], [810, 1080], 0.35)

S_Printer = School.add_appliance(1, 20, 2, 30, 0.1, 5, name="printer")
S_Printer.windows([510, 750], [810, 1080], 0.35)

S_Freezer = School.add_appliance(1, 200, 1, 1440, 0, 30, "yes", 3, name="freezer")
S_Freezer.windows([0, 1440])
S_Freezer.specific_cycle_1(200, 20, 5, 10)
S_Freezer.specific_cycle_2(200, 15, 5, 15)
S_Freezer.specific_cycle_3(200, 10, 5, 20)
S_Freezer.cycle_behaviour(
    [580, 1200], [0, 0], [510, 579], [0, 0], [0, 509], [1201, 1440]
)

S_TV = School.add_appliance(1, 60, 2, 120, 0.1, 5, occasional_use=0.5, name="TV")
S_TV.windows([510, 750], [810, 1080], 0.35)

S_DVD = School.add_appliance(1, 8, 2, 120, 0.1, 5, occasional_use=0.5, name="DVD")
S_DVD.windows([510, 750], [810, 1080], 0.35)

S_Stereo = School.add_appliance(
    1, 150, 2, 90, 0.1, 5, occasional_use=0.33, name="stereo"
)
S_Stereo.windows([510, 750], [810, 1080], 0.35)


uc = UseCase(
    users=User_list,
    parallel_processing=False,
)
uc.initialize(num_days=n_days, peak_enlarge=0.15)

Profiles_list = uc.generate_daily_load_profiles(flat=False)

Profiles_avg, Profiles_list_kW, Profiles_series = pp.Profile_formatting(Profiles_list)

# pp.Profile_series_plot(Profiles_series)  # by default, profiles are plotted as a series

# if (
#     len(Profiles_list) > 1
# ):  # if more than one daily profile is generated, also cloud plots are shown
#     pp.Profile_cloud_plot(Profiles_list, Profiles_avg)

filename_RAMP = "Load_Profiles_total_average_hourly.csv"
Profiles_avg_kW = np.array(Profiles_avg)/1000
Profiles_series_kW = np.array(Profiles_series)/1000

Profiles_avg_df = pd.DataFrame(Profiles_avg_kW)
profiles_index = pd.date_range(start=startdate, periods=len(Profiles_avg_df), freq="min")
Profiles_avg_df.index = profiles_index
Profiles_avg_df_hourly = Profiles_avg_df.resample("h").mean()
Profiles_avg_df_hourly.to_csv("Load_Profiles_daily_average_hourly.csv", index=True)

Profiles_series_df = pd.DataFrame(Profiles_series_kW)
profiles_index_series = pd.date_range(start=startdate, periods=len(Profiles_series_df), freq="min")
Profiles_series_df.index = profiles_index_series
Profiles_series_df_hourly = Profiles_series_df.resample("h").mean()
Profiles_series_df_hourly.to_csv("Load_Profiles_total_average_hourly.csv", index=True)

RAMP_total_load_profile_hourly_kW = pd.read_csv(filename_RAMP, index_col=0, parse_dates=True)

# Convert the hourly load profile to a list for PyPSA 
if isinstance(RAMP_total_load_profile_hourly_kW, pd.Series):
    RAMP_total_load_profile_list = RAMP_total_load_profile_hourly_kW.tolist()
else:
    RAMP_total_load_profile_list = RAMP_total_load_profile_hourly_kW.iloc[:, 0].tolist()

# --------------------------
# Integration of renewables.ninja for pv generation profiles
# --------------------------

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

# --------------------------
# Integration of pylpg for load profiles
# --------------------------

if os.path.exists(filename_LPG): # Load data if it already exists
    # Load the CSV file into a DataFrame
    community_load_profile = pd.read_csv(filename_LPG, index_col=0, parse_dates=True)
    print("Loaded existing load profile from CSV.")
else: # Generate load profile if it does not already exist
    print("File not found. Generating load profile...")
    # Generate load profiles
    community_profiles = {
    name: lpg_execution.execute_lpg_single_household(2024, household, house_type)["Electricity_HH1"].resample("h").sum()
    for name, (household, house_type) in {
        "Single_Person": (lpgdata.Households.CHR07_Single_with_work, lpgdata.HouseTypes.HT20_Single_Family_House_no_heating_cooling),
        "Family": (lpgdata.Households.CHR27_Family_both_at_work_2_children, lpgdata.HouseTypes.HT18_Normal_House_with_15_000_kWh_Gas_Heating_and_a_hot_water_storage_tank),
        "Retired_Couple": (lpgdata.Households.CHR54_Retired_Couple_no_work, lpgdata.HouseTypes.HT08_Normal_house_with_15_000_kWh_Heating_and_5_000_kWh_Cooling_Continuous_Flow_Electric_Heat_Pump)
    }.items()
    }

    # Convert to DataFrame and save
    community_load_profile = pd.DataFrame(community_profiles)
    community_load_profile.to_csv("community_load_profile.csv")

    print("Community load profile generated and saved as CSV.")

total_load_curve = community_load_profile.sum(axis=1)  # Sum across all columns (households)

# Only take the required amount of data points
load_profile_hourly = total_load_curve.iloc[:n_hours]
# Convert the hourly load profile to a list for PyPSA 
if isinstance(load_profile_hourly, pd.Series):
    LPG_load_profile_list = load_profile_hourly.tolist()
else:
    LPG_load_profile_list = load_profile_hourly.iloc[:, 0].tolist()

# Piecewise addition of the two load profiles from RAMP and LPG
total_load = (np.array(RAMP_total_load_profile_list) + np.array(LPG_load_profile_list)).tolist()

# print("Resampled hourly load profile (first 50 values):", load_profile_list[:50])

# --------------------------
# PyPSA Model Setup
# --------------------------

# Create a new PyPSA network
network = pypsa.Network()

# Define time snapshots (n_hours, hourly resolution)
snapshots = pd.date_range(startdate, periods=n_hours, freq="h")
network.set_snapshots(snapshots)

# Add an electricity carrier
network.add("Carrier", "electricity")

# Add buses (one for main, one for PV, one for battery)
network.add("Bus", "main_bus", carrier="electricity")
network.add("Bus", "pv_bus", carrier="electricity")
network.add("Bus", "battery_bus", carrier="electricity")

# Add the load from pylpg
network.add("Load", "household_load", bus="main_bus", p_set=total_load)

# # Add a constant load
# network.add("Load", "Constant_Load", bus="main_bus", p_set=[20] * n_hours)

# Add a PV generator
# Generate a PV profile over one day and  repeat it for n_days
# Here, the sinusoidal profile represents the maximum per-unit PV output
# pv_profile_daily_sinusoidal = [max(0, np.sin(np.pi * h / 24)) for h in range(24)]
# pv_profile = pv_profile_daily_sinusoidal * n_days  # Repeat for each day

# Align the PV profile with PyPSA snapshots
# pv_profile = pv_profile_RenewablesNinja.reindex(network.snapshots, method="nearest")  # Match closest timestamps
network.add("Generator", "pv_generator",
            bus="pv_bus",
            carrier="electricity",
            control="PQ",
            p_nom_extendable=True,
            capital_cost=100,  # €/kW
            marginal_cost=0,
            p_nom=100,  # Initial PV size (kW)
            curtailment_rate_max=0,
            p_max_pu=pv_profile_list)

# Define battery parameters
e_nom_fixed = 10  # kWh
initial_state_of_charge = 0.8

# Add a Store to represent the battery storage
network.add("Store", "battery_storage",
            bus="battery_bus",
            carrier="electricity",
            e_nom=e_nom_fixed,  # Storage capacity (kWh)
            e_nom_extendable=True,
            capital_cost=300,  # €/kWh
            marginal_cost=0,
            e_initial=e_nom_fixed * initial_state_of_charge,  # Initial stored energy (kWh)
            e_cyclic=True)

# Charging Link: from pv_bus to battery_bus
network.add("Link", "battery_charge",
            bus0="pv_bus",
            bus1="battery_bus",
            carrier="electricity",
            marginal_cost=0,
            efficiency=0.8,  # Charging efficiency
            p_nom=15,        # Maximum charging power (kW)
            p_nom_extendable=True)

# Discharging Link: from battery_bus to main_bus
network.add("Link", "battery_discharge",
            bus0="battery_bus",
            bus1="main_bus",
            carrier="electricity",
            marginal_cost=0,
            efficiency=0.8,  # Discharging efficiency
            p_nom=40,        # Maximum discharging power (kW)
            p_nom_extendable=True)

# Direct Link: from pv_bus to main_bus (PV directly serving the load)
network.add("Link", "pv_to_load",
            bus0="pv_bus",
            bus1="main_bus",
            carrier="electricity",
            marginal_cost=0,
            efficiency=1.0,
            p_nom=40,
            p_nom_extendable=True)

#  Custom Constraint 
def define_cons_max_curtailment(network, generator_name):
    """
    Define the constraint for the maximum curtailment rate of a specified generator.

    Args:
    network (Network): The network object containing generators and other components.
    generator_name (str): The name of the generator for which the constraint is to be defined.

    The function adds a constraint to the network model that limits the maximum curtailment rate
    for the specified generator. The curtailment rate is taken from the 'curtailment_rate_max' column
    in the network's generator DataFrame.
    """
    model = network.model
    c_rate_max = network.generators.loc[generator_name, "curtailment_rate_max"]
    model.add_constraints(
        model.variables["Generator-p"].sel(Generator=generator_name).sum()
        >= (
            (1 - c_rate_max)
            * network.generators_t.p_max_pu[generator_name].sum()
            * model.variables["Generator-p_nom"][generator_name].to_linexpr()
        ),
        name=f"{generator_name}_max_curtailment",
    )

network.optimize.create_model() # Create the model
define_cons_max_curtailment(network, generator_name="pv_generator") # Attach the constraint before solving
print(network.model) # print all variables and constrains used in the model
network.optimize.solve_model() # Solve model


# Print optimal values
print("Optimized PV Capacity (kW):", network.generators.p_nom_opt["pv_generator"])
print("Battery Capacity (kWh):", network.stores.e_nom_opt["battery_storage"])
print("Direct link to consumer (kW):", network.links.p_nom_opt["pv_to_load"])
print("Link to Charge (kW):", network.links.p_nom_opt["battery_charge"])
print("Link to Discharge (kW):", network.links.p_nom_opt["battery_discharge"])



# Extract results for plotting
battery_soc = network.stores_t.e["battery_storage"]
load_profile_res = network.loads_t.p["household_load"]
pv_generation_res = network.generators_t.p["pv_generator"]
battery_charge_res = network.links_t.p0["battery_charge"]
battery_discharge_res = network.links_t.p0["battery_discharge"]
direct = network.links_t.p0["pv_to_load"]

# Fetch optimized values
optimized_pv = network.generators.p_nom_opt["pv_generator"]
optimized_battery = network.stores.e_nom_opt["battery_storage"]
optimized_direct_link = network.links.p_nom_opt["pv_to_load"]
optimized_battery_charge = network.links.p_nom_opt["battery_charge"]
optimized_battery_discharge = network.links.p_nom_opt["battery_discharge"]

# Create figure with two subplots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

markersize = 5
linewidth_dashed = 2
linewidth_dotted = linewidth_dashed+1
# --- Subplot 1: Load, PV Generation, and Battery Interaction ---
axes[0].plot(network.snapshots, load_profile_res, label="Load (kW)", color="red", linestyle="-", marker="x", markersize = markersize)
axes[0].plot(network.snapshots, pv_generation_res, label="PV Generation (kW)", color="green", linestyle="--", linewidth = linewidth_dashed)
axes[0].plot(network.snapshots, direct, label="Direct to Load (kW)", color="brown", marker="v", markersize = markersize)
axes[0].plot(network.snapshots, -battery_charge_res, label="Battery Charging (kW)", linestyle="dotted", color="purple", linewidth = linewidth_dotted)
axes[0].plot(network.snapshots, battery_discharge_res, label="Battery Discharging (kW)", linestyle="dotted", color="orange", linewidth = linewidth_dotted)

axes[0].set_title("Power Flow in the Microgrid")
axes[0].set_ylabel("Power (kW)")
axes[0].legend()
axes[0].grid()

# --- Subplot 2: Battery SoC ---
axes[1].plot(network.snapshots, battery_soc, label="Battery SoC (kWh)", color="blue", marker="o", markersize = markersize)

if network.stores.e_nom_opt.empty:
    battery_capacity = network.stores.e_nom["battery_storage"]
else:
    battery_capacity = network.stores.e_nom_opt["battery_storage"]
axes[1].axhline(y=battery_capacity, color="gray", linestyle="--", label="Battery Capacity (kWh)")

axes[1].set_title("Battery State of Charge")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Energy (kWh)")
axes[1].legend()
axes[1].grid()

# --- Add Textbox with Optimized Values ---
textstr = (
    f"Optimized PV Capacity: {optimized_pv:.2f} kW\n"
    f"Battery Capacity: {optimized_battery:.2f} kWh\n"
    f"Direct Link Capacity: {optimized_direct_link:.2f} kW\n"
    f"Battery Charge Capacity: {optimized_battery_charge:.2f} kW\n"
    f"Battery Discharge Capacity: {optimized_battery_discharge:.2f} kW"
)

# Add textbox in the first subplot
props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
axes[0].text(0.02, 0.98, textstr, transform=axes[0].transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

# --- Final Adjustments ---
plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
plt.tight_layout()
plt.show()