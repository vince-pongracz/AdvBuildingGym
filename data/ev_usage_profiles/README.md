# EV User Profiles

This directory contains CSV files that describe electric vehicle (EV) charging session profiles for use in building energy simulations. Each file represents a charging point (e.g. `ev_1.csv` for charging point 1) and contains a time-ordered sequence of EV connection events.

## CSV Format

Each row represents either an EV **connection** or **disconnection** event at the charging point:

- **Connection row**: All fields are populated. A new EV (or the same EV returning) begins a charging session.
- **Disconnection row**: Only `start` has a value; all other fields are null/empty. This marks the moment the EV is unplugged and the charger becomes idle.

Connection and disconnection rows alternate in pairs. The charger is idle between a disconnection and the next connection.

| Field | Type | Unit | Description |
|---|---|---|---|
| `start` | datetime | — | Timestamp when this EV is connected to the charger. A new row means a new charging session begins. |
| `max_cap_kWh` | float | kWh | Total battery capacity of the connected EV. |
| `max_charging_kW` | float | kW | Maximum AC charging power the EV's onboard charger can accept. This is limited by both the vehicle and the wallbox/EVSE. |
| `charger_efficiency` | float | — | Efficiency of grid-to-battery charging (0–1). Accounts for AC/DC conversion and cable losses. |
| `discharge_efficiency` | float | — | Efficiency of battery-to-grid discharging (0–1). Only relevant when `v2g_enabled` is `True`. |
| `v2g_enabled` | bool | — | Whether this EV supports vehicle-to-grid (V2G) operation, i.e. feeding energy back to the building or grid. |
| `start_soc` | float | — | State of charge (0–1) of the EV battery at the moment of connection. |
| `target_soc` | float | — | Desired state of charge (0–1) the user wants when they disconnect. |
| `target_soc_reach_duration_h` | float | h | Time window in which the `target_soc` should be reached, if physically feasible given the battery size and charging power. The optimizer may freely schedule charging within this window. |

## EV Specifications Used

The sample file `ev_1.csv` models three different vehicles that connect to the same charging point over time:

| Vehicle | Battery Capacity | Max AC Charging | Notes |
|---|---|---|---|
| Nissan Leaf (Gen 2) | 40 kWh | 6.6 kW | Affordable city EV, no V2G support in this profile |
| VW ID.4 Pro | 77 kWh | 11 kW | Mid-size SUV, V2G-capable |
| Tesla Model 3 LR | 75 kWh | 11 kW | Popular long-range sedan, V2G-capable |

## Usage Notes

- **Sign convention**: Positive power values represent charging (grid → battery). Negative values (when V2G is active) represent discharging (battery → grid).
- **SOC values** are normalized to 0–1, where 1.0 = fully charged.
- **Charging feasibility**: If the energy required to go from `start_soc` to `target_soc` exceeds what can be delivered at `max_charging_kW` within `target_soc_reach_duration_h`, the system charges at maximum power and the target may not be fully met.
- **Adding new profiles**: Create additional CSV files (e.g. `ev_2.csv`, `ev_3.csv`) following the same format to model multiple charging points.
