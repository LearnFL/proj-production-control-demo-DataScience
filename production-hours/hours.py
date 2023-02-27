import pandas as pd
from pathlib import Path
import math
import jobCatalog as operations
from loadWorkBook import WorkBook
from dataFrame import DataController

class Hours(WorkBook, DataController):
    def __init__(self):
        super().__init__()
        self.orderSize = None
        self.hours = 0
        self.product = None
        self.ml_Hours = 0
        self.ts_Hours = 0
        self.sw_Hours = 0
        self.sl_Hours = 0
        self.lap_Hours = 0
        self.rnd_Hours = 0
        self._totalHours = 0
        
    def _calculateHours(self, orderSize, product, month):
        try:
            self.orderSize = int(orderSize) 
        except:
            self.orderSize = 0

        try:
            self.product = product.strip().lower()
        except:
            self.product = 'no job'
        self.hours = 0

        try:
            for key, value in operations.catalog.get(self.product).items():
                if isinstance(value, list):
                    self._minQuantity(key, value, month, minQuantity=True)
                if not isinstance(value, list):
                    self._minQuantity(key, value, month, minQuantity=False)
        except:
            self.hours = 0

        return self.hours

    def _minQuantity(self, machine, value, month, minQuantity=False):
        curHours = 0
        try:
            if minQuantity ==  True:
                if value[0] > value[1] * self.orderSize:
                    curHours = math.ceil(value[0])

                elif value[0] <= value[1] * self.orderSize:
                    curHours = math.ceil(value[1] * self.orderSize)
            
            if minQuantity == False:
                curHours = math.ceil(value * self.orderSize)

            self.hours += curHours

            match machine:
                case 'ts':
                    self.ts_Hours += curHours 
                case 'mllr':
                    self.ml_Hours += curHours 
                case 'sw':
                    self.sw_Hours += curHours 
                case 'rnd':
                    self.rnd_Hours += curHours 
                case 'sl':
                    self.sl_Hours += curHours 
                case 'lap':
                    self.lap_Hours += curHours 

            self.machineHoursPerMonth(month, machine, curHours)

        except:
            self.hours += 0

    def renderHours(self):
        self.prepWorkBook()

        for index, row in self.df.iterrows():
            hourPerJob = self._calculateHours(row['QUANTITY'], row['ORDER'], row['DUE'])
            row['HOURS'] = hourPerJob
            self.ws1[f'L{index+9}'] = hourPerJob

        total_hours = self._calcTotalHours()
        self._machineHoursBreakdown(total_hours)

    def _hours_Col_Itter(self):
        for col in self.ws1['L9:L300']:
            for cell in col:
                try:
                    yield cell.value
                except:
                    yield 0

    def _calcTotalHours(self):
        total_hours = 0
        for hours in self._hours_Col_Itter():
            if hours:
                total_hours += hours
        return math.ceil(total_hours)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__!r}, Used to calculate job hours, {self.orderSize!r}'

