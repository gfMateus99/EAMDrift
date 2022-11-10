# -*- coding: utf-8 -*-
"""

Models abstract class

"""

from abc import ABC, abstractmethod

class ModelsInterface(ABC):
    
    @abstractmethod
    def __init__(self, ola):
        pass

    @abstractmethod
    def getModelName(self):
        pass
    
    @abstractmethod
    def gridSearch(self):
        pass
    
    @abstractmethod
    def run_and_fit_model(self):
        pass

    @abstractmethod
    def predict(self):
        pass
     
