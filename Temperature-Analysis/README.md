# Project Title

This project is a software that reads, analyzes and plots large amounts of contentious data from sensors. It has been designed to mimic the original project with all proprietary data removed.

## Description

This software is designed to read and analyze large amounts of contentious data from sensors. All accumilated data is later used with machine learning to perform regression analysis and growth predictions. The software is capable of plotting data against run plans loaded from YAML files. 

## Performance

This software uses class polymorphism and ProcessPoolExecuter to read, load, and combine logs in parallel. This enhances the performance and efficiency of the software, enabling it to handle large datasets. In a previous use case, it successfully analyzed 24 million rows of data across 6 columns.

## Warning

Please note that using Pairplots (sns.pairplot) may consume a significant amount of resources and time when used with overly large datasets.


## Authors

This software has been created my Dennis Rotnov.

## Project Status

The project is fully functional and is work in progress. There is a plan to make a GUI for added convenience. 

|File|Description|
|---|---|
|ac.py|File where you set up data|
|error.py|Custom errors|
|helper.py|Helper functions and run dictionary with temperatures|
|loader.py|Loads, reads, performs initial cleaning of excel files|
|plotting.py|Functions for custom plotting|
|redflag.py|Function that handles red flag logging|
|run_plan.yaml|Contains run plan temperature that will be used to analyse data|
