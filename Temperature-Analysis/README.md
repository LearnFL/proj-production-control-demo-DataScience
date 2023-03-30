# Cleans, analyse large volume of continuous data.
#### Performance
I used it to analyse up to 24m rows of data with 6 collumns.<br>
Pairplots (sns.pairplot) may consume alot of resources and time when used with <b>overly</b> large data. 

#### Product related dictionary with temperature data
Either use dictionary or YAML. Possibly YAML is more efficient due to the fact that python does need to build and store in memory potentially large dictionary.

|File|Description|
|---|---|
|ac.py|File where you set up data|
|error.py|Custom errors|
|helper.py|Helper functions and run dictionary with temperatures|
|loader.py|Loads, reads, performs initial cleaning of excel files|
|plotting.py|Functions for custom plotting|
|redflag.py|Function that handles red flag logging|
|run_plan.yaml|Contains run plan temperature that will be used to analyse data|
