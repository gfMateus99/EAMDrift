# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 15:34:56 2022

@author: b28069
"""




class ModelsDB:
    def __init__(self, train_df_: object, pointsToPredict_: int, columnToPredict_: str, time_column_: str):
        self.pointsToPredict = pointsToPredict_
        self.train_df = train_df_
        self.columnToPredict = columnToPredict_
        self.time_column = time_column_

    def run_models(self, model_to_run):
        
        if model_to_run == "ExponentialSmoothing":
            from EAMDrift_model.ModelsDB.ExponentialSmoothingClass import ExponentialSmoothingClass
            return ExponentialSmoothingClass(model_to_run, self.train_df, self.pointsToPredict, self.columnToPredict, self.time_column)  
        
        #elif model_to_run == "YOUR_MODEL_NAME":
            #from EAMDrift_model.ModelsDB.YOUR_MODEL_NAME_CLASS import YOUR_MODEL_NAME_CLASS
            #return YOUR_MODEL_NAME_CLASS(model_to_run, self.train_df, self.pointsToPredict, self.columnToPredict, self.time_column)  
        
        
        else:
            raise Exception("Model is not implemented")
        
