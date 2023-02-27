'''
Web crawler that looks for certain article titles that meet criterea/subjects.
Before use must make sure that crawling/scrapping is allowed by the website owner/admin.
'''

from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from bs4 import BeautifulSoup
import re
import random
import os
from time import sleep
import platform 

class ArticleCrawler:
    def __init__(self, domain, delay):
        self.reverse_counter = 0
        self.visitedList = set()
        self.index_offset = 0
        self.delay = delay
        self.domain = domain

    def filter(self, x):
        for i in filterList:
            if  i in x.lower() :
                return x
    
    def platform(self):
        try:
            if platform.system() == 'Windows':
                return os.path.expanduser('~\Desktop\web_scraping.txt')
            else:
                return os.path.expanduser('~/Desktop/web_scraping.txt')
        except:
            print('OS could not be determined')
            raise

    def getPage(self, link):
        try:
            html = urlopen(f'{self.domain}/{link}')
        except HTTPError as e:
            print(e)
            return None
        except URLError as e:
            print(e)
            return None
        try:
            bs = BeautifulSoup(html, 'lxml')
        except:
            print('Beautifulsoup could not get info')
        else:
            return bs

    def safeGet(self, pageObject, selector):
        self.reverse_counter = len(self.visitedList) - self.index_offset

        try:
            search_result = random.choice(pageObject.findAll(selector, text=self.filter , href=compile))
            return search_result
            
        except AttributeError as e:
            print(e)
            if len(self.visitedList) > 0:
                print('Attribute Error, grabbing a previous link.')
                self.getPage(list(self.visitedList)[self.reverse_counter])
                self.index_offset += 1        
            else:
                return None

    def parse(self, link, selector):  
        bs = self.getPage(link)
        if bs is not None:
            search = self.safeGet(bs, selector)
            opener = open (self.platform(), 'a')
            try:
                if (search['href'] not in self.visitedList):
                    self.visitedList.add(search['href'])
                    with opener:
                        opener.write(search['title'] + '\n')
                        opener.write(search['href'] + '\n')
                    print(search['title'])
                    print(search['href'])
                    print('_'*30)
                    sleep(self.delay)
                    self.parse(search['href'])
                else:
                    sleep(self.delay)
                    self.parse(search['href'])
            finally:    
                opener.close()

filterList = [
        'interest1', 
        'interest2', 
        'interest3',
        ..........
]

parser = ArticleCrawler(domain='SOME_DOMAIN', delay=5)        
parser.parse(link='', selector='a')
