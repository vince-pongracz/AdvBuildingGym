# State Sources

State sources provide observation signals to the building environment. They are split into two categories: inner and outer.

It's worth mentioning, that some states are introduced by the infrastructure elements, so not all state space variable is defined in a statesource.

## Inner

State sources that describe or modify the environment's **internal state**, affected by agent actions.

- `BuildingHeatLoss` — models heat transfer between inside and outside, modifying indoor temperature

## Outer

State sources that represent **external constraints or signals**, driven by time and independent of agent actions.

- `InsideTemperature` — desired indoor temperature setpoint
- `WeatherDataSource` — outdoor temperature from weather data
- `EnergyPriceYearDynDataSource` — energy market price signal
- `EVState` — EV connect/disconnect schedule (from CSV)
- `OperatorEnergyControl` — grid operator power limits
- `DesiredUserEnergyNeed` — user energy consumption profile

---

TODO VP: maybe the above mentioned distinction does not make sense at all...