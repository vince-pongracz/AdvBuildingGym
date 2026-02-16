# State Sources

State sources provide observation signals to the building environment. They are split into two categories:

## Inner

State sources that describe or modify the environment's **internal state**, affected by agent actions.

- `BuildingHeatLoss` — models heat transfer between inside and outside, modifying indoor temperature

## Outer

State sources that represent **external constraints or signals**, driven by time and independent of agent actions.

- `InsideTemperature` — desired indoor temperature setpoint
- `WeatherDataSource` — outdoor temperature from weather data
- `EnergyPriceDataSource` — energy market price signal
- `EVState` — EV connect/disconnect schedule (from CSV)
- `OperatorEnergyControl` — grid operator power limits
- `DesiredUserEnergyNeed` — user energy consumption profile
