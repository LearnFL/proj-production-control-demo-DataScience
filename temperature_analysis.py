'''
THIS SOFTWARE HAS BEEN DESIGNED AND BUILT BY DENNIS ROTNOV
ALL RIGHTS RESERVED
How to use:
    -   Update file path.
    -   Update start/end date.
    -   Update step (step will skip rows while reading files).
    -   Update redFlagValue, it sets temperature diviation level for detecting abnormal/sudden temperature changes.
    -   Update run type [RUN_NAME].
    -   Update collumns (which collumns to read).
    -   Update parquet_name (save dataframe as..).
Recommendations:
    -   Check Flagged.csv to find abnormal temperature varientions.
Watch:
    -   If run plan temperature range is not plotted that is because you did not enter correct start and finish dates.
'''

import pandas as pd
from datetime import datetime, timedelta
from pandas.core.indexes.base import Index
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from pathlib import Path
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from matplotlib.ticker import MaxNLocator
import numpy as np
import seaborn as sns
import os
from os import error, path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm
import csv
from threading import Thread

# File Location                                                                                                         
file_path=Path(r"FILE_PATH/FILE_NAME")                  

# Run set up                                                                                                                         
start_date=datetime(2022,3,10) # YYYY,MM,DD                                                                     
end_date=datetime(2022,9,25)    # YYY,MM,DD
step = 1
redFlagValue = 2                                                                                 
run_type = "NAME_OF_PRODUCT_TO_RETRIEVE_VALUE_FROM_DICT"                                                                                                                                      
col = [0,1,7,9,11]
parquet_name = 'NAME_OF_FILE'

# Temperature filter STR for csv
low_diss= 10  
heigh_diss= 40  
low_top = 31
heigh_top = 40
low_td = -1
heigh_td = 7

# Custom Error class with default message that could be overrriden
class Error(Exception):
    pass

class RedFlagDictKeyNotFound(Error):
    def __init__(self, message="Key for the Red Flag Dictionary is not found."):
        self.message = message
        super().__init__(self.message)


class PathError(Error):
    def __init__(self, message="Check file path"):
        self.message = message
        super().__init__(self.message)

class DataFrameEmpty(Error):
     def __init__(self, message="Data Frame is empty\nMake sure you entered correct start and finish dates"):
        self.message = message
        super().__init__(self.message)

class RunTypeError(Error):
    def __init__(self, message="Please make sure you entered correct run type"):
        self.message = message
        super().__init__(self.message)

class CodeError(Error):
    def __init__(self, message="Ask admin for help."):
        self.message = message
        super().__init__(self.message)

# Checking path
def pathCheck(file_path,*args):
    if path.exists(file_path):
        print('Path exists')
    else:
        raise PathError(*args)

pathCheck(file_path)

# Determening file type
def readerSelector(file_path):
    ext =''
    root_dir, tail = path.split(path.realpath(os.listdir(file_path)[0]))
    extensions = [".xls", ".csv", ".xlsx"]
    for j in extensions:
        if Path(tail).suffix == j:
            ext = j
    if ext == ".csv":
        reader = pd.read_csv
    elif ext == ".xls" or ext == ".xlsx":   
        reader = pd.read_excel
    else:
        raise PathError('Must supply CSV, XLS or XLXS files')
    return reader, ext
        
reader, ext = readerSelector(file_path)
print (f'File extension: {ext}, Reader: {reader}')

li=[]

def read(file): 
    df = reader(file, usecols=col, skiprows=lambda x: x in range(1, -1, step), dtype='object', parse_dates=True)
    df=df.iloc[41::]
    print(file)
    print(df.head(5))
    yield df

def toList(i):
    li.append(read(i))
   
def execute():
    threads = [Thread(target=toList(i)) for i in Path(file_path).rglob("*" + ext)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()

execute()

print('Comibining\n')
all_files = pd.concat((next(i) for i in li))

print('Creating columns\n')
all_files.columns=["Date","Time","Diss","Top", "TD"]
print(all_files)
print('Creating DateTime Index\n')

# Creating DateTime index
all_files['DateTime'] =pd.to_datetime(all_files['Date'] + ' ' + all_files['Time']) #infer_datetime_format=True
all_files = all_files.set_index('DateTime',drop=False)
df = all_files
print(df.head(5))

# Dropping unused columns
print('Dropping columns\n')
drop=['Date','Time']
df= all_files.drop(columns=drop)

print('Concatination and Indexing have finished.')

# Checking if DataFrame is empty
def frameCheck(df, *args):
    if df.empty == True:
        raise DataFrameEmpty(*args)
    else:
        print('DataFrame is not empty\n')

frameCheck(df)

# Removing empty cells and NaN values. Using multiple methods to make sure it is done.
pd.to_numeric(df['Diss'], errors='coerce')
pd.to_numeric(df['Top'], errors='coerce')
pd.to_numeric(df['TD'], errors='coerce')
df['Diss'].replace('', np.nan, inplace=True)
df['Top'].replace('', np.nan, inplace=True)
df['TD'].replace('', np.nan, inplace=True)
df.dropna(inplace=True)
df=df[df[['Diss', 'Top','TD', 'DateTime']] != " "] 

df=df.astype({'Diss': 'float64','Top':'float64','TD':'float64','DateTime': 'datetime64[ns]'})
print(df.dtypes)

# Sorting Data Frame
print('Sorting Data Frame by Date and Time\n')
df.sort_index(inplace=True)

df.info()
print('\n')
print(df.head(5))
print('Removing data recorded when thermocouples were unplugged \n')

# Removing data when TC was disconnected. Typically it shows as a very high or low temperatures, well beyond our limits.
drop_diss_low=df[df["Diss"]<low_diss].index
df.drop(drop_diss_low,inplace=True, errors='ignore') 
drop_top_low=df[df["Top"]<low_top].index
df.drop(drop_top_low, inplace=True, errors='ignore')
drop_diss_high=df[df["Diss"]>heigh_diss].index
drop_top_high=df[df["Top"]>heigh_top].index
df.drop(drop_diss_high,inplace=True, errors='ignore') 
df.drop(drop_top_high, inplace=True, errors='ignore')
drop_td_low=df[df["TD"]<low_td].index
df.drop(drop_td_low,inplace=True, errors='ignore')
drop_td_high=df[df["TD"]>heigh_td].index
df.drop(drop_td_high,inplace=True, errors='ignore') 

# Discards the first and the last days of the run that may contain temperatures that should not be taken into calculations
print('Trtimming Beginning and end of DataFrame \n')
df.drop(df.loc[df['DateTime'] >(end_date-timedelta(days=1))].index, inplace=True)
df.drop(df.loc[df['DateTime'] <= (start_date-timedelta(days=-1))].index, inplace=True)

if not frameCheck(df, 'Lost data after removing first/last day or unplugged thermocouple signal.\nCheck entered shutdown date and/or temperature filter'):
    print('Saving Data Frame\n')
    df.to_parquet(os.path.join(file_path, f'{parquet_name}.df.parquet.gzip'),
                compression='gzip')  

''' ------------------------------------THIS SECTION WRITES STATISTICS------------------------------------------------'''

frameCheck(df, 'Lost data after saving the Data Frame')

print('Writing Custom statistics...\n')
st=df.copy()
st=df.drop(columns="DateTime")
st=st.astype(float)
print(st.dtypes)
print('\n')
st=st.agg(
            {
                "Diss":['mean','std','skew','kurt'],
                "Top":['mean','std','skew','kurt'],
                "TD":['mean','std','skew','kurt'],
            }
           )
st.to_csv(file_path /'Stats.csv') 

print('Writing Red Flag temperatures...\n')
frameCheck(df, "Data Frame should not be empty, needs debugging.")
df = df.astype({'DateTime': 'datetime64[ns]', 'Diss': 'float64','Top':'float64','TD':'float64'})

# Run Plan Dictionary
# Temperature dictionary. All proprietary data has been removed.
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

redFlag = run_plan.get(run_type, RedFlagDictKeyNotFound(message="Cannot retreave run plan from Red Flag Dictionary, make sure you entered correct run plan."))

# Red Flag settings
redFlagDict = {
    "td_max": [x + redFlagValue for x in redFlag.get('td', RedFlagDictKeyNotFound(message="The run has been found, but values are not retriavable. Failed reading td_max."))],
    "td_min": [x - redFlagValue for x in redFlag.get('td', RedFlagDictKeyNotFound(message="The run has been found, but values are not retriavable. Failed reading td_min."))],
    "diss_max": [x + redFlagValue for x in redFlag.get('diss', RedFlagDictKeyNotFound(message="The run has been found, but values are not retriavable. Failed reading diis_max."))],
    "diss_min": [x - redFlagValue for x in redFlag.get('diss', RedFlagDictKeyNotFound(message="The run has been found, but values are not retriavable. Failed reading diss_min."))],
    "top_max": [x + redFlagValue for x in redFlag.get('top', RedFlagDictKeyNotFound(message="The run has been found, but values are not retriavable. Failed reading top_max."))],
    "top_min": [x - redFlagValue for x in redFlag.get('top', RedFlagDictKeyNotFound(message="The run has been found, but values are not retriavable. Failed reading top_min."))]
}

df = df.astype({'DateTime': 'datetime64[ns]', 'Diss': 'float64','Top':'float64','TD':'float64'})

# Creates date ranges for temperature ramp plot
a10 = df[df.DateTime.between((start_date - timedelta(days=-1)),(start_date - timedelta(days=-10)))].copy()
a20 = df[df.DateTime.between((start_date - timedelta(days=-10)),(start_date - timedelta(days=-20)))].copy()
a30 = df[df.DateTime.between((start_date - timedelta(days=-20)),(start_date - timedelta(days=-30)))].copy()
a40 = df[df.DateTime.between((start_date - timedelta(days=-30)),(start_date - timedelta(days=-40)))].copy()
a50 = df[df.DateTime.between((start_date - timedelta(days=-40)),(start_date - timedelta(days=-50)))].copy()
a60 = df[df.DateTime.between((start_date - timedelta(days=-50)),(start_date - timedelta(days=-60)))].copy()
a70 = df[df.DateTime.between((start_date - timedelta(days=-60)),(start_date - timedelta(days=-70)))].copy()
a80 = df[df.DateTime.between((start_date - timedelta(days=-70)),(start_date - timedelta(days=-80)))].copy()

listOfRamp = [a10, a20, a30, a40, a50, a60, a70, a80]
df_flagged = pd.DataFrame(columns=["DateTime", 'Diss', 'Top', 'TD']).set_index('DateTime',drop=False)

def redFlagLogger(*, ramp, tempDict, dataFrame):
    error_set = set() # For debugging, catches unique errors and exceptions.  
    red_flag_set =set() # To quickly visualize during which period a Red Flag was found and in which zone.
    periodList = ["010 days", "020 days", "030 days","040 days","050 days", "060 days", "070 days", "080 days"]
    index_for_period = 0
    for r in ramp:
            for i, j, k in zip(['TD', 'Top', 'Diss'],['td_max', 'top_max', 'diss_max'], ['td_min', 'top_min', 'diss_min']):
                t = 0
                for t in range(0, len(list(tempDict[j]))):
                    try:
                        r.loc[((r[i]>tempDict[j][t]) | (r[i]<tempDict[k][t])),"Flag"] = 1
                        if r.empty != True:
                            red_flag_set.add(f'{periodList[index_for_period]} period in {i}')   
                        dataFrame = pd.concat([dataFrame, r])
                        t+=1 
                    except Exception as e:
                        error_set.add(str(e))
                        continue
            index_for_period += 1        
    print(f'List of Exceptions:  {error_set}')    
    print(f'List of detected Red Flag: {sorted(red_flag_set, key=(lambda x: x[0:4]))}')          
    dataFrame.to_csv(file_path /'Flagged.csv') 

redFlagLogger(ramp=listOfRamp, tempDict=redFlagDict, dataFrame=df_flagged)

print('Finished writing Red Flag\n')
frameCheck(df, 'Lost Data after writing Red Flag' )

# Writing statistics for Night vs Day temperature fluctuations
print('Writing Nights Vs Days\n')
night=df.copy()
night=night.between_time("23:00","5:00")
night=night[['Diss','Top','TD']].astype(float)
print(night.dtypes)
st_night= night.agg(
            {
              "Diss":['mean','std','skew','kurt'],
              "Top":['mean','std','skew','kurt'],
              "TD":['mean','std','skew','kurt'],
            })   
st_night.to_csv(file_path /'Nights.csv') 

day=df.copy()
day=day.between_time('5:01',"22:59")
day=day[['Diss','Top','TD']].astype(float)
print(day.dtypes)
print('\n')
st_day= day.agg(
            {
            "Diss":['mean','std','skew','kurt'],
            "Top":['mean','std','skew','kurt'],
            "TD":['mean','std','skew','kurt'],
            }) 
st_day.to_csv(file_path /'Days.csv') 

frameCheck(df, 'Lost data after writing Nights and Days')

# Writing statistics for Week days
print('Writing Week days\n')
df_wd=df.copy()
df_wd[df_wd.index.weekday<5]
df_wd=df_wd[['Diss','Top','TD']].astype(float)
print(df_wd.dtypes)
print('\n')
df_wd_stat=df_wd.agg(
            {
            "Diss":['mean','std','skew','kurt'],
            "Top":['mean','std','skew','kurt'],
            "TD":['mean','std','skew','kurt'],
            })  
df_wd_stat.to_csv(file_path /'Week days.csv') 

frameCheck(df, 'Lost data after writing Days')

# Writing statistics for Weekend
print('Writing Weekends\n')
df_we=df.copy()
df_we[df_we.index.weekday>4]
df_we=df_we[['Diss','Top','TD']].astype(float)
print(df_we.dtypes)
print('\n')
df_we_stat=df_we.agg(
            {
            "Diss":['mean','std','skew','kurt'],
            "Top":['mean','std','skew','kurt'],
            "TD":['mean','std','skew','kurt'],
            })  
df_we_stat.to_csv(file_path /'Weekend.csv') 

frameCheck(df, 'Lost data after writing Weekends')
print(df.head(5))

print('Starting Regression\n')

# To average values every X rows enter a different value into min_periods
rdf = df.copy()
rdf.Diss = rdf.Diss.rolling(1, min_periods=1).mean()
rdf.Top = rdf.Top.rolling(1, min_periods=1).mean()
rdf.TD = rdf.TD.rolling(1, min_periods=1).mean()
rdf=rdf.iloc[::1]
rdf.dropna(inplace=True)
print(rdf.info())
rdf.drop(columns='DateTime')

# Prints short regression report
print('Diss to Top Regression\n')
x = rdf.Diss.values.reshape((-1, 1))
y = rdf.Top.values
model = LinearRegression()
model.fit(x, y)
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')
export={'Coef of Determination':r_sq, 'Intercept':model.intercept_, 'Slope': model.coef_, 'Predicted Responce':y_pred}
w = csv.writer(open(file_path/"Diss-Top.csv", "w"))
for key, val in export.items():
    w.writerow([key, val])

print('TD to Top Regression\n')
x = rdf.TD.values.reshape((-1, 1))
y = rdf.Top.values
model = LinearRegression()
model.fit(x, y)
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')
export={'Coef of Determination':r_sq, 'Intercept':model.intercept_, 'Slope': model.coef_, 'Predicted Responce':y_pred}
w = csv.writer(open(file_path/"TD-Top.csv", "w"))
for key, val in export.items():
    w.writerow([key, val])

print('Top to TD Regression\n')
x = rdf.Top.values.reshape((-1, 1))
y = rdf.TD.values
model = LinearRegression()
model.fit(x, y)
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')
export={'Coef of Determination':r_sq, 'Intercept':model.intercept_, 'Slope': model.coef_, 'Predicted Responce':y_pred}
w = csv.writer(open(file_path/"Top-TD.csv", "w"))
for key, val in export.items():
    w.writerow([key, val])

# Prints Full Regression report
print('Advances Regression Diss to Top')
x = rdf.Diss.values.reshape((-1, 1))
y = rdf.Top.values
x = sm.add_constant(x)
model = sm.OLS(y, x)
results = model.fit()
print(results.summary())
export=[results.summary()]
with open(file_path/'Adv Diss-Top.txt','w') as data: 
      data.write(str(export))

print('Advances Regression Diss to TD')
x = rdf.Diss.values.reshape((-1, 1))
y = rdf.TD.values
x = sm.add_constant(x)
model = sm.OLS(y, x)
results = model.fit()
print(results.summary())
export=[results.summary()]
with open(file_path/'Adv Diss-TD.txt','w') as data: 
      data.write(str(export))

print('Advances Regression TD to Top')
x = rdf.TD.values.reshape((-1, 1))
y = rdf.Top.values
x = sm.add_constant(x)
model = sm.OLS(y, x)
results = model.fit()
print(results.summary())
export=[results.summary()]
with open(file_path/'Adv TD-Top.txt','w') as data: 
      data.write(str(export))

print('Advances Regression Top to TD')
x = rdf.Top.values.reshape((-1, 1))
y = rdf.TD.values
x = sm.add_constant(x)
model = sm.OLS(y, x)
results = model.fit()
print(results.summary())
export=[results.summary()]
with open(file_path/'Adv Top-Td.txt','w') as data: 
      data.write(str(export))

print('Writing Statistics Finished!\n')
print('Creating Charts....\n')

frameCheck(df, 'Lost data after writing regressions')
   
if (i := run_plan[run_type.lower().strip()]) is None:
    raise RunTypeError
else:
    try:
        td_limit = i.get('td')
        diss_limit = i.get('diss')
        top_limit = i.get('top')
    except:
        raise CodeError(message="Run type has been recognized but the values for Top and/or Main and/or TD are not retrivable.") 

print(f"TD: {td_limit}")
print(f"Diss: {diss_limit}")
print(f"Top: {top_limit}")
print("\n")
print("Reading\n")

'''--------------------------------THIS AREA IS RESPONSIBLE FOR BUILDING PLOTS------------------------------------------'''                           

# Functions to plot Ramp lines for each date period, takes X that could be either td_limit or diss_limit or top_limit
def plot10(x:float):
    a = plt.plot([a10["DateTime"].values[1],a10["DateTime"].values[-1]], [x[0] - 0.5, x[1] - 0.5] , 'k:',color="r", lw=1) 
    b = plt.plot([a10["DateTime"].values[1],a10["DateTime"].values[-1]], [x[0] + 0.5, x[1] + 0.5], 'k:',color="r", lw=1) 
    return a, b

def plot20(x:float):
    a = plt.plot([a20["DateTime"].values[1],a20["DateTime"].values[-1]], [x[1] - 0.5, x[2] - 0.5] , 'k:',color="r", lw=1) 
    b = plt.plot([a20["DateTime"].values[1],a20["DateTime"].values[-1]], [x[1] + 0.5, x[2] + 0.5], 'k:',color="r", lw=1) 
    return a, b

def plot30(x:float):
    a = plt.plot([a30["DateTime"].values[1],a30["DateTime"].values[-1]], [x[2] - 0.5, x[3] - 0.5] , 'k:',color="r", lw=1) 
    b = plt.plot([a30["DateTime"].values[1],a30["DateTime"].values[-1]], [x[2] + 0.5, x[3] + 0.5], 'k:',color="r", lw=1) 
    return a, b

def plot40(x:float):
    a = plt.plot([a40["DateTime"].values[1],a40["DateTime"].values[-1]], [x[3] - 0.5, x[4] - 0.5] , 'k:',color="r", lw=1) 
    b = plt.plot([a40["DateTime"].values[1],a40["DateTime"].values[-1]], [x[3] + 0.5, x[4] + 0.5], 'k:',color="r", lw=1) 
    return a, b

def plot50(x:float):   
    a = plt.plot([a50["DateTime"].values[1],a50["DateTime"].values[-1]], [x[4] - 0.5, x[5] - 0.5] , 'k:',color="r", lw=1) 
    b = plt.plot([a50["DateTime"].values[1],a50["DateTime"].values[-1]], [x[4] + 0.5, x[5] + 0.5], 'k:',color="r", lw=1) 
    return a, b

def plot60(x:float):
    a = plt.plot([a60["DateTime"].values[1],a60["DateTime"].values[-1]], [x[5] - 0.5, x[6] - 0.5] , 'k:',color="r", lw=1) 
    b = plt.plot([a60["DateTime"].values[1],a60["DateTime"].values[-1]], [x[5] + 0.5, x[6] + 0.5], 'k:',color="r", lw=1) 
    return a, b
        
def plot70(x:float):
    a = plt.plot([a70["DateTime"].values[1],a70["DateTime"].values[-1]], [x[6] - 0.5, x[7] - 0.5] , 'k:',color="r", lw=1) 
    b = plt.plot([a70["DateTime"].values[1],a70["DateTime"].values[-1]], [x[6] + 0.5, x[7] + 0.5], 'k:',color="r", lw=1) 
    return a, b

def plot80(x:float):
    a = plt.plot([a80["DateTime"].values[1],a80["DateTime"].values[-1]], [x[7] - 0.5, x[8] - 0.5] , 'k:',color="r", lw=1) 
    b = plt.plot([a80["DateTime"].values[1],a80["DateTime"].values[-1]], [x[7] + 0.5, x[8] + 0.5], 'k:',color="r", lw=1) 
    return a, b

func_list = [plot10, plot20, plot30, plot40, plot50, plot60, plot70, plot80]

class Plotter:
    def __init__(self, func_list):
        self.func_list = func_list
    
    def __call__(self, limit):
        for func in self.func_list:
            try:
                func(limit)
            except:
                pass 

    def __repr__(self) -> str:
        return f'{self.__class__.__name__!r}, Used to plot temperature range for graphs, {func_list!r}'

plotter = Plotter(func_list)
   
# Plot chart for TD
plt.plot(df["DateTime"], df["TD"])
plt.title("TD")
plt.ylabel("TD")
plt.xlabel("Date")
plotter(td_limit)
plt.savefig(os.path.join(file_path, 'TD.png'))

# Plot chart for Diss
df[['DateTime','Diss']].plot.scatter(y="Diss", x="DateTime", alpha=0.5)
plt.title("Diss")
plt.ylabel("Diss")
plt.xlabel("Date")
plotter(diss_limit)
plt.savefig(os.path.join(file_path, 'Diss.png'))

# Plot chart for Top
df[['DateTime','Top']].plot.scatter(y="Top", x="DateTime", alpha=0.5)
plt.title("Top")
plt.ylabel("Top")
plt.xlabel("Date")
plotter(top_limit)
plt.savefig(os.path.join(file_path, 'Top.png'))

plt.subplot(2,2,1)

# Creating a Distribution histogram for Top
plt.hist(df['Top'], edgecolor="grey")
plt.title("Top")
plt.subplot(2,2,2)

# Creating a Distribution histogram for Diss
plt.hist(df['Diss'], edgecolor="grey")
plt.title("Diss")
plt.subplot(2,2,3)

# Creating a Distribution histogram for TD
plt.hist(df['TD'], edgecolor="grey")
plt.title("TD")
plt.savefig(os.path.join(file_path, 'Dist.png'))

# Creating density charts
plt.subplot(2,2,1)
sns.distplot(df['Diss'], bins=10)
plt.subplot(2,2,2)
sns.distplot(df['Top'], bins=10)
plt.subplot(2,2,3)
sns.distplot(df['TD'], bins=10)
plt.savefig(os.path.join(file_path, 'Density.png'))

# Creating regression plots
plt.figure(figsize=(10,8))
sns.regplot(x='Diss',y="Top", data=rdf[['Diss', 'Top']]) #rdf
plt.xlabel("Diss")
plt.ylabel("Top")
plt.savefig(os.path.join(file_path, 'Reg Diss-Top.png'))

# Creating regression plots
plt.figure(figsize=(10,8))
sns.regplot(x='Diss',y="TD", data=rdf[['Diss', 'TD']]) #rdf
plt.xlabel("Diss")
plt.ylabel("TD")
plt.savefig(os.path.join(file_path, 'Reg Diss-TD.png'))

# Creating regression plots
plt.figure(figsize=(10,8))
sns.regplot(x='TD',y="Top", data=rdf[['TD', 'Top']]) #rdf
plt.xlabel("TD")
plt.ylabel("Top")
plt.savefig(os.path.join(file_path, 'Reg TD-Top.png'))

# Creating Day and Night charts
day["D/N"]="Day"
night["D/N"]="Night"
snsdata = pd.concat([day, night], join="outer" )
snsdata.drop(columns='TD')
snsdata.sort_index(ascending=True)
g=sns.pairplot(data=snsdata, hue="D/N", plot_kws={'alpha':0.4})
for ax in g.axes.flat:
    ax.xaxis.set_major_locator(MaxNLocator(5, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(5, prune="both"))
plt.savefig(os.path.join(file_path, 'Day-Night.png'))

g=sns.pairplot(data=df[['TD','Top','Diss']], plot_kws={'alpha':0.4})
for ax in g.axes.flat:
    ax.xaxis.set_major_locator(MaxNLocator(5, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(5, prune="both"))
plt.savefig(os.path.join(file_path, 'Pairplot.png'))

print("Finished!")
