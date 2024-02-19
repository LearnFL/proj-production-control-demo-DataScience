# Production Schedule Calculator

This repository contains a Python-based software tool used to calculate labor and machine hours for production scheduling. The software is designed to read, process and write back data to Excel, providing useful calculations for total time per job, time per machine, and per operation. 

## Features

- The software uses the `pandas` module to read from Excel and grab data such as part numbers, order size, and due dates.
- It performs various calculations and writes the data back to Excel using `openpyxl`.
- Calculations include total time for jobs, time per machine, per operation.
- The software also takes into account the minimum time needed for a job/machine, regardless of order size.
- If the minimum time required is more than the total time per order per operation, the minimum time is used in calculations.
- In addition to the above, the software breaks down machine time by machine type per month.

### FILES
#### jobCatalog.py
Contains part numbers with minimum hours and multipliers per part.

#### loadWorkBook.py
Handles loading of Excel file and writes data.

#### hours.py
Performs calculations

#### dataFRame.py
Handles Pandas DataFrame

#### main.py
Initializes work.

