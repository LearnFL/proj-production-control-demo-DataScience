# Cleans, analyse large volume of continuous data.
#### Performance
I used it to analyse up to 24m rows of data with 6 collumns.<br>
Pairplots (sns.pairplot) may consume alot of resources and time when used with <b>overly</b> large data. 

|File|Description|
|---|---|
|ac.py|File where you set up data|
|error.py|Custom errors|
|helper.py|Helper functions and run dictionary with temperatures|
|loader.py|Loads, reads, performs initial cleaning of excel files|
|plotting.py|Functions for custom plotting|
|redflag.py|Function that handles red flag logging|
