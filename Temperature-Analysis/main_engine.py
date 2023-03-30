
import pandas as pd
from datetime import timedelta
from pandas.core.indexes.base import Index
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import csv
from time import time

from error import RedFlagDictKeyNotFound, RunTypeError, CodeError
from loader import df_builder
from helper import pathCheck, frameCheck, run_plan
from plotting import Plotter
from redflag import redFlagLogger

def engine(file_path=None, start_date=None, end_date=None, step=None, redFlagValue=None, run_type=None, col=None, parquet_name=None, low_diss=None, heigh_diss=None, low_top=None, heigh_top=None, low_td=None, heigh_td=None, rol_period=None):

    start = time()
    # Checking path
    pathCheck(file_path)

    print('Building\n')

    df = df_builder(file_path, col, step)

    print('Creating columns\n')

    df.columns = ["Date","Time","Diss","Top", "TD"]

    frameCheck(df, 'Lost data after creating columns')

    print('Creating DateTime Index\n')
    # Creating DateTime index
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time']) #infer_datetime_format=True
    df = df.set_index('DateTime',drop=False)
    print(df.head(5))

    # Dropping unused columns
    print('Dropping columns\n')
    drop =['Date','Time']
    df = df.drop(columns=drop)

    print('Concatination and Indexing have finished.')

    frameCheck(df, 'Lost data after dropping Date and atime columns.')


    # BLOCK CLEAN UP
    # Removing empty cells and NaN values. Using multiple methods to make sure it is done.
    pd.to_numeric(df['Diss'], errors='coerce')
    pd.to_numeric(df['Top'], errors='coerce')
    pd.to_numeric(df['TD'], errors='coerce')
    df['Diss'].replace('', np.nan, inplace=True)
    df['Top'].replace('', np.nan, inplace=True)
    df['TD'].replace('', np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df[df[['Diss', 'Top','TD', 'DateTime']] != " "] 

    df = df.astype({'Diss': 'float64','Top':'float64','TD':'float64','DateTime': 'datetime64[ns]'})
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


    #FIXME 
    '''
    Known issue with parquet related to set_index(drop=False) or reset_idex(drop=False)
    Workarround:
    drop DateTime before saving, and readd the column after done saving 
    '''
    if not frameCheck(df, 'Lost data after removing first/last day or unplugged thermocouple signal.\nCheck entered shutdown date and/or temperature filter'):
        print('Saving Data Frame\n')
        df.drop(['DateTime'], axis=1,inplace=True)
        df.to_parquet(os.path.join(file_path, f'{parquet_name}.df.parquet.gzip'),
                    compression='gzip')  
        df['DateTime'] = df.index


    #BLOCK
    ''' ------------------------------------THIS SECTION WRITES STATISTICS------------------------------------------------'''

    frameCheck(df, 'Lost data after saving the Data Frame')

    print('Writing Custom statistics...\n')
    st = df.copy()
    st = df.drop(columns="DateTime")
    st = st.astype(float)
    print(st.dtypes)
    print('\n')
    st = st.agg(
                {
                    "Diss":['mean','std','skew','kurt'],
                    "Top":['mean','std','skew','kurt'],
                    "TD":['mean','std','skew','kurt'],
                }
            )

    st.to_csv(file_path /'Stats.csv') 

    del st

    print('Writing Red Flag temperatures...\n')

    frameCheck(df, "Data Frame should not be empty, needs debugging.")

    df = df.astype({'DateTime': 'datetime64[ns]', 'Diss': 'float64','Top':'float64','TD':'float64'})

    redFlag = run_plan.get(run_type.lower(), RedFlagDictKeyNotFound(message="Cannot retreave run plan from Red Flag Dictionary, make sure you entered correct run plan."))

    # Red Flag settings
    message = "The run has been found, but values are not retriavable. Failed reading td_max"
    redFlagDict = {
        "td_max": [x + redFlagValue for x in redFlag.get('td', RedFlagDictKeyNotFound(message=message))],
        "td_min": [x - redFlagValue for x in redFlag.get('td', RedFlagDictKeyNotFound(message=message))],
        "diss_max": [x + redFlagValue for x in redFlag.get('diss', RedFlagDictKeyNotFound(message=message))],
        "diss_min": [x - redFlagValue for x in redFlag.get('diss', RedFlagDictKeyNotFound(message=message))],
        "top_max": [x + redFlagValue for x in redFlag.get('top', RedFlagDictKeyNotFound(message=message))],
        "top_min": [x - redFlagValue for x in redFlag.get('top', RedFlagDictKeyNotFound(message=message))]
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

    redFlagLogger(ramp=listOfRamp, tempDict=redFlagDict, dataFrame=df_flagged, file_path=file_path)

    print('Finished writing Red Flag\n')
    frameCheck(df, 'Lost Data after writing Red Flag' )

    # Writing statistics for Night vs Day temperature fluctuations
    print('Writing Nights Vs Days\n')
    night = df.copy()
    night = night.between_time("23:00","5:00")
    night = night[['Diss','Top','TD']].astype(float)
    print(night.dtypes)
    st_night= night.agg(
                {
                "Diss":['mean','std','skew','kurt'],
                "Top":['mean','std','skew','kurt'],
                "TD":['mean','std','skew','kurt'],
                })   
    st_night.to_csv(file_path /'Nights.csv') 

    del st_night

    day = df.copy()
    day = day.between_time('5:01',"22:59")
    day = day[['Diss','Top','TD']].astype(float)
    print(day.dtypes)
    print('\n')
    st_day= day.agg(
                {
                "Diss":['mean','std','skew','kurt'],
                "Top":['mean','std','skew','kurt'],
                "TD":['mean','std','skew','kurt'],
                }) 
    st_day.to_csv(file_path /'Days.csv') 

    del st_day

    frameCheck(df, 'Lost data after writing Nights and Days')

    # Writing statistics for Week days
    print('Writing Week days\n')
    df_wd = df.copy()
    df_wd[df_wd.index.weekday<5]
    df_wd = df_wd[['Diss','Top','TD']].astype(float)
    print(df_wd.dtypes)
    print('\n')
    df_wd_stat=df_wd.agg(
                {
                "Diss":['mean','std','skew','kurt'],
                "Top":['mean','std','skew','kurt'],
                "TD":['mean','std','skew','kurt'],
                })  
    df_wd_stat.to_csv(file_path /'Week days.csv') 

    del df_wd
    del df_wd_stat

    frameCheck(df, 'Lost data after writing Days')

    # Writing statistics for Weekend
    print('Writing Weekends\n')
    df_we = df.copy()
    df_we[df_we.index.weekday>4]
    df_we = df_we[['Diss','Top','TD']].astype(float)
    print(df_we.dtypes)
    print('\n')
    df_we_stat=df_we.agg(
                {
                "Diss":['mean','std','skew','kurt'],
                "Top":['mean','std','skew','kurt'],
                "TD":['mean','std','skew','kurt'],
                })  
    df_we_stat.to_csv(file_path /'Weekend.csv') 

    del df_we
    del df_we_stat

    frameCheck(df, 'Lost data after writing Weekends')
    print(df.head(5))

    print('Starting Regression\n')

    # To average values every X rows enter a different value into min_periods
    rdf = df.copy()
    rdf.Diss = rdf.Diss.rolling(1, min_periods=rol_period).mean()
    rdf.Top = rdf.Top.rolling(1, min_periods=rol_period).mean()
    rdf.TD = rdf.TD.rolling(1, min_periods=rol_period).mean()
    rdf = rdf.iloc[::1]
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
    export = {'Coef of Determination':r_sq, 'Intercept':model.intercept_, 'Slope': model.coef_, 'Predicted Responce':y_pred}
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
    export = {'Coef of Determination':r_sq, 'Intercept':model.intercept_, 'Slope': model.coef_, 'Predicted Responce':y_pred}
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
    export = {'Coef of Determination':r_sq, 'Intercept':model.intercept_, 'Slope': model.coef_, 'Predicted Responce':y_pred}
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
    export = [results.summary()]
    with open(file_path/'Adv Diss-Top.txt','w') as data: 
        data.write(str(export))

    print('Advances Regression Diss to TD')
    x = rdf.Diss.values.reshape((-1, 1))
    y = rdf.TD.values
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()
    print(results.summary())
    export = [results.summary()]
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


    # BLOCK
    '''--------------------------------THIS AREA IS RESPONSIBLE FOR BUILDING PLOTS------------------------------------------'''                           

    period = [a10, a20, a30, a40, a50, a60, a70, a80]
    plotter = Plotter(period)
    
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
    sns.displot(df['Diss'], bins=10)
    plt.subplot(2,2,2)
    sns.displot(df['Top'], bins=10)
    plt.subplot(2,2,3)
    sns.displot(df['TD'], bins=10)
    plt.savefig(os.path.join(file_path, 'Density.png'))

    # Creating regression plots
    plt.figure(figsize=(10,8))
    sns.regplot(x='SOMETHING',y="SOMETHING", data=rdf[['SOMETHING', 'SOMETHING']]) #rdf
    plt.xlabel("SOMETHING")
    plt.ylabel("SOMETHING")
    plt.savefig(os.path.join(file_path, 'SOMETHING.png'))

    # Creating regression plots
    plt.figure(figsize=(10,8))
    sns.regplot(x='SOMETHING',y="SOMETHING", data=rdf[['SOMETHING', 'SOMETHING']]) #rdf
    plt.xlabel("SOMETHING")
    plt.ylabel("SOMETHING")
    plt.savefig(os.path.join(file_path, 'SOMETHING.png'))

    # Creating regression plots
    plt.figure(figsize=(10,8))
    sns.regplot(x='SOMETHING',y="SOMETHING", data=rdf[['SOMETHING', 'SOMETHING']]) #rdf
    plt.xlabel("SOMETHING")
    plt.ylabel("SOMETHING")
    plt.savefig(os.path.join(file_path, 'SOMETHING.png'))

    del rdf

    # Creating Day and Night charts
    day["D/N"] = "Day"
    night["D/N"] = "Night"
    snsdata = pd.concat([day, night], join="outer" )
    snsdata.drop(columns='TD')
    snsdata.sort_index(ascending=True)
    g = sns.pairplot(data=snsdata, hue="D/N", plot_kws={'alpha':0.4})
    for ax in g.axes.flat:
        ax.xaxis.set_major_locator(MaxNLocator(5, prune="both"))
        ax.yaxis.set_major_locator(MaxNLocator(5, prune="both"))
    plt.savefig(os.path.join(file_path, 'Day-Night.png'))

    g = sns.pairplot(data=df[['TD','Top','Diss']], plot_kws={'alpha':0.4})
    for ax in g.axes.flat:
        ax.xaxis.set_major_locator(MaxNLocator(5, prune="both"))
        ax.yaxis.set_major_locator(MaxNLocator(5, prune="both"))
    plt.savefig(os.path.join(file_path, 'Pairplot.png'))

    del day
    del night
    del df

    end = time()
    print(f"Finished in {str(timedelta(seconds=(end-start)))}")