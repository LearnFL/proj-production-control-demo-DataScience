'''
This software cleans production data and calculates scrap.
Licence Key implementation is basic and not intendent to substetute propper implementation of a license key.
Dennis Rotnov
'''

from tkinter import ttk
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import filedialog as fd
from pathlib import Path
import pandas as pd
import datetime as dt
import os
from tkinter import messagebox as mb
from tkinter import *
from time import sleep
import math
import pandas as pd

license_flag = 0

class Error(Exception):
    pass

class EmployeeFileNotFound(Error):
    def __init__(self, message="Employee file not found. Make sure the file is in the same folder with the license key"):
        self.message = message
        super().__init__(self.message)

class SaveLiceseKeyError(Error):
    def __init__(self, message="Error while saving license key"):
        self.message = message
        super().__init__(self.message)

class EmployeeIterError(Error):
    def __init__(self, message="Error while iterrating over Employees df"):
        self.message = message
        super().__init__(self.message)        

# WINDOW SETUP

root = tk.Tk()
root.title('\u00a9 YOUR NAME')
root.geometry('500x400+100+50')
root.resizable(False, False)

def save_license (path, l_key, win2):
    if not os.path.exists(path):
        os.mkdir(path)
    try:
        with open(os.path.join(path, 'license-key.txt'), 'w') as writer:
            writer.write(l_key)
            win2.destroy()
    except:
        raise SaveLiceseKeyError

def enterLicense(path):
    win2 = Toplevel()
    win2.title('License')
    win2.geometry("250x200")
    L1 = tk.Label(win2, text = "License Key")
    L1.grid(column=1, columnspan=4, pady=20, padx=90, sticky=tk.W)
    E1 = Entry(win2, bd =5)
    E1.grid(column=1, columnspan=2, padx=60, sticky=tk.W)
    button = ttk.Button(win2, text='Submit', command=lambda: save_license(path, E1.get(), win2))
    button.grid(column=1, padx=85, pady=20, sticky=tk.W) 
        
def checkLicense (path, _date=None):
    license_flag = 0   

    if not os.path.exists(path): 
        license_flag = 0
        enterLicense(path)  
    else:
        try: 
            with open(path+'/license-key.txt', 'r') as f:
                l_key = f.readline()
                a, b, c, d, e, f = l_key.split('-')

                if _date == None:
                    current_date = dt.datetime.today()

                a = 'YOUR_LICENSE_LOGIC'
                b = 'YOUR_LICENSE_LOGIC'
                c = 'YOUR_LICENSE_LOGIC'
                print(a, b ,c)
                
                if ('YOUR_LICENSE_LOGIC'):
                    license_flag = 1
                else:
                    license_flag = 0
                    mb.showerror(title="License", message="License Error.")
                    enterLicense(path)  
        except:
            mb.showerror(title="Error", message="Error. Reenter license key. If the problem persists contact YOUR NAME.")
            enterLicense(path)
    return license_flag
    
def employeesDataFrame():
    # Excell document with all employees names and employee numbers. This is the easiest and the fastest way to allow employee list modifications.
    path = os.path.expanduser('~\employees.xlsx')
    df = pd.read_excel(path)
    df.columns = ['code', 'name']
    EMPLOYEES = {}
    try:
        for idx, row in df.iterrows():
            EMPLOYEES[row['code']] = row['name']
    except:
        raise EmployeeIterError

    return EMPLOYEES

def numberToname(x):
    person = employeesDataFrame()
    try:
        name = person.get(x, 'NO NAME')
    except KeyError:
        name = x
    return name

def upload(_date=None):
    path = r'~\Documents\Scrap'
    path = os.path.expanduser(path)

    flag = checkLicense(path)
   
    if flag == 1:
        try:
            file_path = askopenfilename(filetypes=[('Excel','*.xlsx')])
            print(Path(file_path))
        except:
            mb.showerror(title = "File Upload Error", message = "There was an error uploading the file. Please close and reopen the program.")

        if file_path is not None:
            file_transaction_path = file_path

            SCRAP_DICT = {
                "code1": 'reason1',
                "code2": 'reason2'
            }

            try:
                col = [0, 2, 3, 4, 7, 8, 9, 10, 12, 14, 15, 20]
                df_transaction = pd.read_excel(file_transaction_path, usecols = col, dtype='object', na_values=[''], parse_dates=True)
                df_transaction=df_transaction.iloc[1:-1]

                print(df_transaction.head(5))
                print(df_transaction.dtypes)
                print('Creating columns\n')

                df_transaction.columns=["JOB", "TRANSACTION_TYPE", "DATE","ITEM","COMPLETED","SCRAP", "OPERATION", "T_TOTAL", "EMPLOYEE", "T_START", "T_FINISH", "MOVED"]

                print(df_transaction.head(5))
                print(df_transaction.tail(5))
                print('Setting Date column')

                df_transaction['DATE'] = pd.to_datetime(df_transaction['DATE']).dt.date

                df_drop = df_transaction[df_transaction.JOB.isna()].index
                df_transaction.drop(df_drop, inplace=True)
                print(df_transaction.head(5))

                df_transaction.JOB = df_transaction.JOB.astype('string')
                
                # df clean up, eliminates extra zeros from job numbers
                def job_cleanup(x):
                    if x[0].upper() == 'P':
                        x = ''.join([x[1],x[5:]]) 
                    elif ((x[0].upper() == "K") or (x[0].upper() == "V")):
                        x = ''.join([x[2:4], x[6:]])
                    else:
                        x = x 
                    return x

                df_transaction.JOB = df_transaction.JOB.apply(job_cleanup)
                print(df_transaction.head(5))

                # Employee number to name
                df_transaction.EMPLOYEE.astype('string')
                
                df_transaction.EMPLOYEE = df_transaction.EMPLOYEE.apply(numberToname)
                print(df_transaction.head(5))

                # Eliminationg Indirect jobs
                df_drop = df_transaction[df_transaction['TRANSACTION_TYPE'] == "Indirect"].index
                df_transaction.drop(df_drop, inplace=True, errors='ignore')
                print(df_transaction.head(5))

                # Scrap
                df_transaction.SCRAP = df_transaction.SCRAP.apply(lambda x: int(x))
                df_drop = df_transaction[df_transaction['SCRAP'] == 0].index
                df_transaction.drop(df_drop, inplace=True, errors='ignore')

                test = df_transaction.copy()

                df_by_job = df_transaction.groupby(['ITEM']).SCRAP.sum()
                df_transaction = df_transaction.groupby(['EMPLOYEE', 'JOB', 'ITEM', 'OPERATION', 'TRANSACTION_TYPE']).SCRAP.sum()
                test = test.agg({
                    "JOB" : ['count'],
                    'SCRAP': ['sum' ],
                    "COMPLETED": ['sum']
                })

            except:
                mb.showinfo(title = "File Error", message = "You have uploaded a file with a wrong layout or the data in columns is corrupted.")                

            try:
                print(df_transaction)
                save_path = os.path.expanduser('~\Desktop\Scrap.xlsx')
            
            except:
                mb.showinfo(title = "Saving Error", message = "Error while saving file on your desktop. Make sure your have permissions to write to the hard drive. Close the program and try again.") 

            try: 

                with pd.ExcelWriter(save_path) as writer:
                    df_transaction.to_excel(writer, sheet_name="Detailed")
                    df_by_job.to_excel(writer, sheet_name="Total Per Item")
                    test.to_excel(writer, sheet_name="Totals")

            except PermissionError:

                mb.showinfo(title = "Permission Error", message = "Scrap.xlsx is open in Excel. Please close the spreadsheet and try again.")                

            else:

                os.startfile(save_path)

                mb.showinfo(title = "Done", message = "Report has been submitted. Excel spreadsheet will open in a new window.")

# MAIN GUI

file_upload_label = ttk.Label(root, text="Please upload trasaction report from InFor.", font=('Arial 14 bold'))
file_upload_label.grid(padx=50, pady=70, sticky=tk.W)
upload_button = ttk.Button(root, text="Upload", command=upload)
upload_button.grid(padx=200,pady=10, sticky=tk.W)

root.mainloop()
