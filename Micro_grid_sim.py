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


# ---------------------------------TODO:------------------
# Integrate RAMP or LoadProfileGenerator
# Integrate renewables.ninja data
# Implement addition of new consumers
# Implement a parser to control the simulation from the "outside"


import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

num_days = 7

# Create a new PyPSA network
network = pypsa.Network()

# Define time snapshots (24 hours * num_days)
snapshots = pd.date_range("2023-01-01", periods=24 * num_days, freq="h")
network.set_snapshots(snapshots)

# Add an electricity carrier
network.add("Carrier", "electricity")

network.add("Bus", "main_bus", carrier="electricity")
network.add("Bus", "pv_bus", carrier="electricity")
network.add("Bus", "battery_bus", carrier="electricity")  # Separate bus for the battery

# Add a constant load of 20 kW
load_profile = [20] * 24 * num_days
network.add("Load", "constant_load", bus="main_bus", p_set=load_profile)

# Add a PV generator
pv_profile = [max(0, np.sin(np.pi * h / 24)) for h in range(24)] * num_days
pv_profile_min = [max(0, 0*h) for h in range(24)] * num_days
network.add("Generator", "pv_generator",
            bus="pv_bus",
            carrier="electricity",
            control = "PQ",
            p_nom_extendable=True,
            capital_cost=100,  # €/kW
            marginal_cost=0,
            p_nom=100,  # Initial PV size
            p_nom_min = 10,
            curtailment_rate_max = 0,
            p_max_pu=pv_profile)


e_nom_fixed= 150
initial_state_of_charge = 0.8
# Add a Store to represent the battery
network.add("Store", "battery_storage",
            bus="battery_bus",
            carrier="electricity",
            e_nom=e_nom_fixed,  # Storage capacity 
            e_nom_max = 1000,
            e_nom_extendable=True,  
            capital_cost=150,  # €/kWh
            marginal_cost = 0,
            e_initial=e_nom_fixed*initial_state_of_charge,  # Initial energy stored
            e_cyclic=True)  # Cyclic SoC

# Add Links for Charging and Discharging the Battery
# Charging Link (Power flows from main_bus to battery_bus)
network.add("Link", "battery_charge",
            bus0="pv_bus",
            bus1="battery_bus",
            carrier="electricity",
            marginal_cost=0,
            efficiency=0.9,  # Charging efficiency
            p_nom=15,  # Charging power 
            p_nom_extendable=False)

# Discharging Link (Power flows from battery_bus to main_bus)
network.add("Link", "battery_discharge",
            bus0="battery_bus",
            bus1="main_bus",
            carrier="electricity",
            marginal_cost=0,
            efficiency=0.9,  # Discharging efficiency
            p_nom=40,  # Discharging power 
            p_nom_extendable=False)

# Direct Link (Power flows from pv_bus to main_bus)
network.add("Link", "pv_to_load",
            bus0="pv_bus",
            bus1="main_bus",
            carrier="electricity",
            marginal_cost=0,
            efficiency=1.0,  
            p_nom=40, 
            p_nom_extendable=False)

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

# Extract results
battery_soc = network.stores_t.e["battery_storage"]
load_profile = network.loads_t.p["constant_load"]
pv_generation = network.generators_t.p["pv_generator"]
battery_charge = network.links_t.p0["battery_charge"]
battery_discharge = network.links_t.p0["battery_discharge"]
direct = network.links_t.p0["pv_to_load"]

# Plot everything
plt.figure(figsize=(12, 6))

# Plot Battery SoC
plt.plot(network.snapshots, battery_soc, label="Battery SoC (kWh)", color="blue", marker="o")

# Plot Load
plt.plot(network.snapshots, load_profile, label="Load (kW)", color="red", linestyle="-", marker="x")

# Plot PV Generation
plt.plot(network.snapshots, pv_generation, label="PV Generation (kW)", color="green", linestyle="--")
plt.plot(network.snapshots, direct, label="Direct to load (kW)", color="brown", marker="v")

# Plot Battery Charge/Discharge
plt.plot(network.snapshots, battery_charge, label="Battery Charging (kW)", linestyle="dotted", color="purple")
plt.plot(network.snapshots, -battery_discharge, label="Battery Discharging (kW)", linestyle="dotted", color="orange")

# Add a line for battery capacity
if network.stores.e_nom_opt.empty:    
    plt.axhline(y=network.stores.e_nom["battery_storage"], color="gray", linestyle="--", label="Battery Capacity (kWh)")
else:
    plt.axhline(y=network.stores.e_nom_opt["battery_storage"], color="gray", linestyle="--", label="Battery Capacity (kWh)")
# Add titles, labels, and legend
plt.title("Microgrid Operation Over Time")
plt.xlabel("Time")
plt.ylabel("Energy or Power (kWh or kW)")
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()

# Show the plot
plt.show()
