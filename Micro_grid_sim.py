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
# adjusted timeframing of simulation to enable renewables.ninja data

# ---------------------------------TODO:------------------
# Integrate RAMP 
# Integrate "Standardlastprofile"
# Implement addition of new consumers
# Implement a parser to control the simulation from the "outside"

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

filename = "community_load_profile.csv"
n_days = 21 # MUST BE GREATER THAN 1
n_hours = 24 * n_days
startdate = "2023-01-01"

# Convert string to datetime object, add n_days and convert it back
startdate_dt = datetime.strptime(startdate, "%Y-%m-%d")
enddate_dt = startdate_dt + timedelta(days=n_days-1)
enddate = enddate_dt.strftime("%Y-%m-%d")

# --------------------------
# Integration of renewables.ninja for pv generation profiles
# --------------------------


token = 'd0c7b389b3a5523e6d4c23c64776e0539ad5e543'
api_base = 'https://www.renewables.ninja/api/'

session = requests.session()
# Send token header with each request
session.headers = {'Authorization': 'Token ' + token}

url = api_base + 'data/pv'

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

if os.path.exists(filename): # Load data if it already exists
    # Load the CSV file into a DataFrame
    community_load_profile = pd.read_csv(filename, index_col=0, parse_dates=True)
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
    load_profile_list = load_profile_hourly.tolist()
else:
    load_profile_list = load_profile_hourly.iloc[:, 0].tolist()

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
network.add("Load", "household_load", bus="main_bus", p_set=load_profile_list)

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
