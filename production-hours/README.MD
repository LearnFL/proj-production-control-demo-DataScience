# HOURS CALCULATER FOR A PRDUCTION SCHEDULE

### DATA
All proprietary data has been removed, data entered in jobCatalog.py is intended to simulate real project.

### DESCRIPTION
Used to calculate labor and machine hours on a production schedule. 
The software uses pandas module to read Excel and grab data like part numbers, order size, due dates, then performs calculations and writes data back into excel using openpyxl. Calculations include total time for all jobs, time per machine, per operation, also it takes into account minimum time needed for a job/machine, regardless the order size. If minimum time required is more than total time per order per operation, minimum time will be used for calculations. In addition to all listed above the software breaks machine time down by machine type and month.

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

