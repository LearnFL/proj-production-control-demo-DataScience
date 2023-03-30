'''
THIS SOFTWARE HAS BEEN DESIGNED AND BUILT BY DENNIS ROTNOV
ALL RIGHTS RESERVED
How to use:
    -   Update file path.
    -   Update start/end date.
    -   Update step (step will skip rows while reading files).
    -   Update redFlagValue, it sets temperature diviation level for detecting abnormal/sudden temperature changes.
    -   Update run type [swept, statek, wp, lpof, saw, systron.......].
    -   Update collumns (which collumns to read).
    -   Update parquet_name (save dataframe as..).
Recommendations:
    -   Check Flagged.csv to find abnormal temperature varientions.
Watch:
    -   If temperature run plan temperature range is not plotted that is because you did not enter correct start and finish dates.
'''

from datetime import datetime
from pathlib import Path
from time import time

from main_engine import engine

# BLOCK SET-UP

# File Location                                                                                                         
file_path=Path(r"FILE_PATH/FILE_NAME")                  
                                                                                                            

# Run set up                                                                                                                         
start_date = datetime(2022,3,10) # YYYY,MM,DD                                                                     
end_date = datetime(2022,9,25)    # YYY,MM,DD
step = 1
redFlagValue = 2                                                                                 
run_type = "NAME_OF_PRODUCT_TO_RETRIEVE_VALUE_FROM_DICT"            
col = [0,1,7,9,11]
rol_period = 1 # Average values in Data Frame every X rows
parquet_name = 'NAME_OF_FILE'

# Temperature filter STR for csv
low_diss = 10  
heigh_diss = 40  
low_top = 31
heigh_top = 40
low_td = -1
heigh_td = 7

engine(file_path, start_date, end_date, step, redFlagValue, run_type, col, parquet_name, low_diss, heigh_diss, low_top, heigh_top, low_td, heigh_td, rol_period)