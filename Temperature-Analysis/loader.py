from pathlib import Path
import os
import pandas as pd
from threading import Thread, Lock

from error import PathError

class Reader:
    def __init__(self, file, col, step):
        self.file= file
        self.col = col
        self.step = step

    def readerSelector(self):
        ext =''
        tail = Path(self.file).suffix.lower()[1:]
        extensions = ["xls", "csv", "xlsx"]
        for j in extensions:
            if tail == j:
                ext = j
        if ext == "csv":
            reader = pd.read_csv
        elif ext == "xls" or ext == "xlsx":   
            reader = pd.read_excel
        else:
            raise PathError('Must supply CSV, XLS or XLXS files')
        return reader, ext

    def read(self): 
        reader, ext = self.readerSelector()
        df = reader(self.file, usecols=self.col, skiprows=lambda x: x in range(1, -1, self.step), dtype='object', parse_dates=True)
        df=df.iloc[41::]
        print(self.file)
        print(df.head(5))
        return df

    @classmethod
    def generate_inputs(cls, col, step, file_dir):
        for file in os.listdir(file_dir):
            if Path(file).suffix.lower()[1:] in ['xlsx', 'xls', 'csv']: 
                yield cls(os.path.join(file_dir, file), col, step)


class Loader:
    def __init__(self, input_data):
        self.input = input_data
        self.lock = Lock()
        self.df = None
    
    def build_df(self):
        self.df = self.input.read()
        return self.df
    
    def concat(self, other):
        self.df = pd.concat([self.df, other.df])
        
    @classmethod
    def generate_workers(cls, input_class, col, step, file_dir):
        workers = []
        for input in input_class.generate_inputs(col, step, file_dir):
            workers.append(cls(input))
        return workers

     
def execute(workers):
    threads = [Thread(target=w.build_df) for w in workers ]
    for thread in threads: thread.start()
    for thread in threads: thread.join()

    first, *rest = workers
    for worker in rest:
        first.concat(worker)
    return first.df


def df_builder(file_dir, col, step, input_class=Reader, worker_class=Loader):
    workers = worker_class.generate_workers(input_class, col, step, file_dir)
    return execute(workers)