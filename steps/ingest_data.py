import logging
from abc import ABC, abstractmethod
import pandas as pd

class IngestData:

    '''
    Ingesting data from the data_path
    
    '''
    def __init__(self, data_path: str):
        self.data_path = data_path

    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        train = pd.read_csv('Data/train.csv')['text'][0]
        val = pd.read_csv('Data/validation.csv')['text'][0]
        test = pd.read_csv('Data/test.csv')['text'][0]
        return train+val+test