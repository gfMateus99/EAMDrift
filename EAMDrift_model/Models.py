# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 15:34:56 2022

@author: b28069
"""

from EAMDrift_model.ModelsDB.ExponentialSmoothingClass import ExponentialSmoothingClass
from EAMDrift_model.ModelsDB.ProphetClass import ProphetClass
from EAMDrift_model.ModelsDB.KalmanForecasterClass import KalmanForecasterClass
from EAMDrift_model.ModelsDB.StatsForecastAutoARIMAClass import StatsForecastAutoARIMAClass
from EAMDrift_model.ModelsDB.LSTMClass import LSTMClass
from EAMDrift_model.ModelsDB.TransformerClass import TransformerClass
from EAMDrift_model.ModelsDB.SARIMAXClass import SARIMAXClass

class ModelsDB:
    def __init__(self, train_df_: object, pointsToPredict_: int, columnToPredict_: str, time_column_: str, dataTimeStep_: str):
        self.pointsToPredict = pointsToPredict_
        self.train_df = train_df_
        self.columnToPredict = columnToPredict_
        self.time_column = time_column_
        self.dataTimeStep = dataTimeStep_

    def run_models(self, model_to_run):
        
        if model_to_run == "ExponentialSmoothing":
            return ExponentialSmoothingClass(model_to_run, self.train_df, self.pointsToPredict, self.columnToPredict, self.time_column, self.dataTimeStep)  
        elif model_to_run == "Prophet":
            return ProphetClass(model_to_run, self.train_df, self.pointsToPredict, self.columnToPredict, self.time_column, self.dataTimeStep)  
        elif model_to_run == "KalmanForecaster":
            return KalmanForecasterClass(model_to_run, self.train_df, self.pointsToPredict, self.columnToPredict, self.time_column, self.dataTimeStep)  
        elif model_to_run == "StatsForecastAutoARIMA":
            return StatsForecastAutoARIMAClass(model_to_run, self.train_df, self.pointsToPredict, self.columnToPredict, self.time_column, self.dataTimeStep)  
        elif model_to_run == "LSTM":
            return LSTMClass(model_to_run, self.train_df, self.pointsToPredict, self.columnToPredict, self.time_column, self.dataTimeStep, [1, 0, 150, 0.001, 'MSE'])  
        elif model_to_run == "LSTM2":
            return LSTMClass(model_to_run, self.train_df, self.pointsToPredict, self.columnToPredict, self.time_column, self.dataTimeStep, [2, 0, 150, 0.001, 'l1'])  
        elif model_to_run == "TRANSFORMER":
            return TransformerClass(model_to_run, self.train_df, self.pointsToPredict, self.columnToPredict, self.time_column, self.dataTimeStep, [0.15, 0.0001, 'MSE'])  
        elif model_to_run == "SARIMAX":
            return SARIMAXClass(model_to_run, self.train_df, self.pointsToPredict, self.columnToPredict, self.time_column, self.dataTimeStep)  
              
        
        
        
        #elif model_to_run == "YOUR_MODEL_NAME":
            #from EAMDrift_model.ModelsDB.YOUR_MODEL_NAME_CLASS import YOUR_MODEL_NAME_CLASS
            #return YOUR_MODEL_NAME_CLASS(model_to_run, self.train_df, self.pointsToPredict, self.columnToPredict, self.time_column)  
        
        
        else:
            raise Exception("Model is not implemented")
        
