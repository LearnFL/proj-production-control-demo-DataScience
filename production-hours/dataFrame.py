import pandas as pd
import os
from pathlib import Path
from collections import defaultdict

class DataController:
    def __init__(self):
        self.path = None
        self.df = None
        self.log = defaultdict(dict)

    def _path_exists(self, path):
        self.path = path
        if not os.path.exists(self.path):
            raise Exception('Invalid file path')
        return self.path

    def get_dataFrame(self, path=None):
        self._path_exists(path)
        df = pd.read_excel(self.path, usecols=[5, 8, 9, 11], skiprows=lambda x: x in range(0, 7), dtype='object')
        df.columns = ['ORDER', 'DUE', 'QUANTITY', 'HOURS']
        df['DUE'] = pd.to_datetime(df['DUE']).dt.month
        self.df = df

    def machineHoursPerMonth(self, month, machine, curHours):
        curMonth = self.log[month]
        if machine in curMonth:
            curMonth[machine] += curHours
        else:
            curMonth[machine] = curHours
            
    def _groupByJobType(self):
        dfGroupped = self.df.groupby('ORDER')['QUANTITY'].sum()
        dfGroupped = dfGroupped.to_frame().reset_index()
        return dfGroupped
