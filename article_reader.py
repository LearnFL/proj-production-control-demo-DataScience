from PyPDF2 import PdfReader
from pathlib import Path
import re
import os
from threading import Thread, Lock

dir_path = Path(r"PATH TO A FOLDER")


class InputClass:
    def __init__(self, pdf_file):
        self.pdf_file = pdf_file
        self.lock = Lock()

    def read(self):
        with self.lock:
            with open(self.pdf_file, 'rb') as f:
                pdf = PdfReader(f)
                pages = pdf.pages
                for idx, page in enumerate(pages, 1):
                    yield (len(pages), idx, self.pdf_file, page)

    @classmethod
    def generate_inputs(cls, data_dir):
        for file in os.listdir(data_dir):
            head, tail = os.path.split(file)
            if Path(tail).suffix == '.pdf':
                yield cls(os.path.join(data_dir, file))


class PdfSearch:
    def __init__(self, input, pattern):
        self.input = input
        self.report = []
        self.lock = Lock()
        self.pattern = pattern


    def dump_txt(self):
        with self.lock:
            with open(os.path.join(dir_path, 'report.txt'), "a") as writer:
                for entry in self.report:
                    writer.write(f'{entry} \n\n')

    def present_data(self, pattern, GrabExtra=False):
        for pagesN, idx, file_name, page in self.input.read():
            result = pattern.finditer(page.extract_text())
            for i in result:
                if GrabExtra == False:
                    print(f'File Name: {file_name}, Page: {idx} out of {pagesN}, {i.group()}')
                    self.report.append(f'File Name: {file_name}, Page: {idx} out of {pagesN}, {i.group()}')

                else:
                    try:
                        print(f'File Name: {file_name}, Page: {idx} out of {pagesN}, {page.extract_text()[i.start()-100 : i.end()+100]}')
                    except:
                        print(f'File Name: {file_name}, Page: {idx} out of {pagesN}, {i.group()}')

        self.dump_txt()

    def filter(self):
        pattern = re.compile(self.pattern, re.IGNORECASE)
        self.present_data(pattern, GrabExtra=False)

    @classmethod
    def build_workers(cls, input_class, data_dir, pattern):
        workers = []
        for input_data in input_class.generate_inputs(data_dir):
            workers.append(cls(input_data, pattern))
        return workers


def execute(workers):
    threads = [Thread(target=w.filter) for w in workers]    
    for thread in threads: thread.start()
    for thread in threads: thread.join()


def search_pdf(*, worker_class, input_class, data_dir, pattern):
    workers = worker_class.build_workers(input_class,data_dir, pattern)
    return execute(workers)


main_subject = 'mobility'
key_words = [
    'hydrogen', 'AlOH', 'H', 'Ca', 'Mg', 'Li', 'alkali', 'metals', 'charge','compensator', 'hole', 'electron', 'OH', 'travel', 'move', 'sweep', 
    'sweeping', 'axis', 'current', 'ion', 'covalent', 'valent', 'and', 'Al'
    ]

pattern = r".*"+f"\\b{main_subject}\\b"+"\s*\w*\s*("+'|'.join([f'\\b{i}\\b' for i in key_words])+").*"

search_pdf(worker_class=PdfSearch, input_class=InputClass, data_dir=dir_path, pattern=pattern)
