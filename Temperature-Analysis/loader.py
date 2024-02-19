from pathlib import Path
import os
from os import path
import pandas as pd
from threading import Thread, Lock
from typing import Tuple, Callable, Optional
from error import PathError
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED, FIRST_EXCEPTION


class Reader:
    """
    This class is used to read data from a set of files in a directory.

    The class takes in a list of columns to read, a step size to skip rows, and a directory path.
    It then iterates over all the files in the directory, checking their extensions.
    If the extension is csv, xls, or xlsx, the class uses pandas to read the file.
    It uses the skiprows argument to skip rows based on the step size, and the usecols argument to select the columns.
    The parse_dates argument is set to True to ensure that the dates are read correctly.

    The class also provides a method to generate input objects, which can be used to create workers.
    The workers are used to read the files in parallel, and the results are concatenated into a single dataframe.
    """
    def __init__(self, file: str, col: list[int], step: int) -> None:
        self.file= file
        self.col = col
        self.step = step

    def readerSelector(self):
        """
        This method selects the appropriate reader based on the file extension.
        It returns a tuple containing the reader function and the file extension.
        """
        ext =''
        tail = Path(self.file).suffix.lower()[1:]
        extensions = ["xls", "csv", "xlsx"]
        for j in extensions:
            if tail == j:
                ext = j
                # print (f'File extension: {ext}')
        if ext == "csv":
            reader = pd.read_csv
        elif ext == "xls" or ext == "xlsx":   
            reader = pd.read_excel
        else:
            raise PathError('Must supply CSV, XLS or XLXS files')
        return reader, ext

    def read(self) -> pd.DataFrame: 
        """
        This method reads the data from the file and returns a pandas dataframe.
        
        Return:
        df (pd.DataFrame): pandas dataframe.
        """
        reader, ext = self.readerSelector()
        df = reader(self.file, usecols=self.col, skiprows=lambda x: x in range(1, -1, self.step), dtype='object', parse_dates=True)
        df=df.iloc[41::]
        print(self.file)
        print(df.head(5))
        return df

    @classmethod
    def generate_inputs(cls, col: list[int], step: int, file_dir: str):
        """
        This classmethod generates input objects for the workers.
        It iterates over all the files in the directory, and returns an input object for each file
        that has a supported extension.
        """
        for file in os.listdir(file_dir):
            if Path(file).suffix.lower()[1:] in ['xlsx', 'xls', 'csv']: 
                yield cls(os.path.join(file_dir, file), col, step)


class Loader:
    """
    This class is used to load data from the input objects created by the Reader class.
    It uses a threading.Lock to ensure that only one thread can access the data at a time.

    The class provides a method to build the dataframe, which is called by the workers.
    The results are concatenated into a single dataframe using pandas.concat.

    The class also provides a method to generate workers, which is used to create input objects for the workers.

    """
    def __init__(self, input_data:Callable) -> None:
        self.input = input_data
        # self.lock: Lock = Lock()
        self.df: Optional[pd.DataFrame] = None
    
    def build_df(self):
        """
        This function is used to read data from the input object.
        It uses the read function of the input object to read the data from the file.
        
        Return:
        self: returns class.
        """
        self.df = self.input.read()
        return self
    
    def concat(self, other) -> None:
        """
        This function concatenates two dataframes.

        Parameters:
        other (pd.DataFrame): The dataframe to concatenate with.

        Returns:
        None
        """
        self.df = pd.concat([self.df, other.df])
        
    @classmethod
    def generate_workers(cls, input_class: Callable, col: list[int], step: int, file_dir: str):
        '''
        This classmethod generates input objects for the workers.
        Arguments: 
        input_class: The input class to use.
        col (list[int]): The list of columns to read.
        step (int): The step size to skip rows.
        file_dir (str): The directory containing the files.

        Returns:
        A list of input objects.
        '''

        workers = []
        for input in input_class.generate_inputs(col, step, file_dir):
            workers.append(cls(input))
        return workers

     
def execute(workers) -> pd.DataFrame:
    with ProcessPoolExecutor(4) as exe:
        futures = [exe.submit(w.build_df) for w in workers]
        done, _ = wait(futures, return_when=ALL_COMPLETED)


    if done:
        first, *rest = [i.result() for i in futures]
        for worker in rest:
            first.concat(worker)
        return first.df

def df_builder(file_dir: str, col: list[int], step: int, input_class: Callable=Reader, worker_class: Callable=Loader) -> Callable:
    workers = worker_class.generate_workers(input_class, col, step, file_dir)
    return execute(workers)

