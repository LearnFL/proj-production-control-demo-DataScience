from os import path
from error import PathError, DataFrameEmpty

# Check if path exists
def pathCheck(file_path,*args):
    if path.exists(file_path):
        print('Path exists')
    else:
        raise PathError(*args)
    
# Checking if DataFrame is empty
def frameCheck(df, *args):
    if df.empty == True:
        raise DataFrameEmpty(*args)
    else:
        print('DataFrame is not empty\n')

''' 
Either use dictionary or YAML
Possibly YAML is more efficient due to the fact that python does need to store in memory potentially large dictionary
'''

def run_plan_loader():
    with open(r"SOME_PATH\run_plan.yaml", 'r') as file:
        run_plan = load(file, Loader=FullLoader)
        return run_plan
    
run_plan = {
        
        'PRODUCT_1': 
            {
                "td" : [0, 0, 0, 0, 0],
                "diss" : [0, 0, 0, 0, 0],
                "top" : [0, 0, 0, 0, 0]
            },

        'PRODUCT_2': 
            {
                "td" : [0 , 0, 0, 0],
                "diss" : [0, 0, 0, 0],
                "top" : [0, 0, 0, 0, 0]
            }
    }
