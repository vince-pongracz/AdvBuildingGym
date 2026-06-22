


# DWD data readme

Air temp link: https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/air_temperature/BESCHREIBUNG_obsgermany_climate_10min_air_temperature_de.pdf

Wind link: https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/wind/BESCHREIBUNG_obsgermany_climate_10min_wind_de.pdf

Sun link: https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/solar/BESCHREIBUNG_obsgermany_climate_10min_solar_de.pdf


Data is always about the last 10 minutes, except the air temp data

---

### Scripts

Scripts are in `preprocessing/weather/dwd/`.

- `dwd_fetch.py` — Downloads and extracts wind, solar, and air_temperature zip archives (historical + recent) for station 04177 from the DWD CDC server. Converts the semicolon-delimited `.txt` files to `.csv`. Can be run standalone to download only.
- `dwd_preprocess.py` — Imports data via `dwd_fetch`, selects relevant columns, merges the three data types on timestamp, renames columns, converts the date to UTC datetime, and drops rows where all measurements are missing (-999). Outputs `merged_04177.csv`.

Merged output columns:

| Original   | Renamed          |
|------------|------------------|
| MESS_DATUM | measure_date     |
| FF_10      | avg_wind_speed   |
| DD_10      | wind_dir         |
| GS_10      | sun_shine        |
| DS_10      | diff_sun_shine   |
| TT_10      | air_temp_2m      |
| RF_10      | rel_humidity     |

---

### Sonne

| Spaltename | Beschreibung                         | Fehlwert | Einheit | Typ    | Format |
|------------|--------------------------------------|----------|---------|--------|--------|
| DS_10      | Summe der diffusen Himmelstrahlung   | -999     | J/cm²   | NUMBER | 9990.0 |
| GS_10      | Summe der Globalstrahlung            | -999     | J/cm²   | NUMBER | 9990.0 |
| SD_10      | Summe der Sonnenscheindauer          | -999     | h       | NUMBER | 90.990 |
| LS_10      | Summe der langwelligen Strahlung     | -999     | J/cm²   | NUMBER | 9990.0 |

About the irradiance types: https://www.dwd.de/DE/leistungen/solarenergie/globalstrahlung.html?nn=446142&lsbId=416798#:~:text=Die%20Globalstrahlung%20ist%20die%20am%20Boden%20von,Sonnenh%C3%B6hen%20von%20mehr%20als%2050%C2%B0%20und%20wolkenlosem

--> Only Globalstrahlung matters

### Wind

| Spaltename | Beschreibung                                              | Fehlwert | Einheit | Typ    | Format |
|------------|-----------------------------------------------------------|----------|---------|--------|--------|
| QN         | Qualitaetsniveau                                          |          |         | NUMBER | 990    |
| SLA_10     | Standardabweichung der lateralen Windgeschwindigkeit      | -999     |         | NUMBER | 990.0  |
| SLO_10     | Standardabweichung der longitudinalen Windgeschwindigkeit | -999     |         | NUMBER | 990.0  |
| FF_10      | Mittlere Windgeschwindigkeit                              | -999     | m/s     | NUMBER | 990.0  |
| DD_10      | Mittlere Windrichtung der vorangegangenen 10 Minuten      | -999     | °       | NUMBER | 990    |

### Air temp

| Spaltename | Beschreibung                | Fehlwert | Einheit | Typ    | Format | Note                                                                  |
|------------|-----------------------------|----------|---------|--------|--------|-----------------------------------------------------------------------|
| QN         | Qualitaetsniveau            |          |         | NUMBER | 990    |                                                                       |
| PP_10      | Luftdruck in Stationshoehe  | -999     | hPa     | NUMBER | 9990.0 |                                                                       |
| TT_10      | Lufttemperatur in 2 m Höhe  | -999     | °C      | NUMBER | 990.0  | instant                                                               |
| TM5_10     | Lufttemperatur in 5 cm Höhe | -999     | °C      | NUMBER | 990.0  | instant                                                               |
| RF_10      | Relative Feuchte            | -999     | %       | NUMBER | 990.0  |                                                                       |
| TD_10      | Taupunkttemperatur          | -999     | °C      | NUMBER | 990.0  | Berechnet aus Lufttemperatur in 2 m Höhe und relativer Feuchtemessung |
