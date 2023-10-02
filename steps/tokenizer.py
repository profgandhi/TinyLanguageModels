import logging
from abc import ABC, abstractmethod
import pandas as pd

class Tokenizer(ABC):

    @abstractmethod
    def encode(self):
        pass

    @abstractmethod
    def decode(self):
        pass

class CharacterTokenizer(Tokenizer):

    def __init__(self,text: str):
        self.vocab = sorted(list(set(text)))
        self.itos = {i:ch for i, ch in enumerate(self.vocab)} #integer to string
        self.stoi = {ch:i for i, ch in enumerate(self.vocab)} #string to integer

    def encode(self,s):
        return [self.stoi[ch] for ch in s]

    def decode(self,s):
        return ''.join([self.itos[i] for i in s])