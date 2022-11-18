# -*- coding: utf-8 -*-
"""
ExponentialSmoothingClass
"""

from EAMDrift_model.ModelsDB.Models_Interface import ModelsInterface

from darts import TimeSeries
from darts.models import StatsForecastAutoARIMA

class StatsForecastAutoARIMAClass(ModelsInterface):
    
    def __init__(self, modelName_: str, train_df_: object, pointsToPredict_: int, columnToPredict_: str, time_column_: str, dataTimeStep_: str):
        self.MODEL_NAME = modelName_
        self.points_to_predict = pointsToPredict_
        self.train_df = train_df_
        
                
        self.mean_y = train_df_[columnToPredict_].mean()
        self.std_y = train_df_[columnToPredict_].std()
        
        new_train_set = train_df_.copy()
        
        new_train_set[columnToPredict_] = (new_train_set[columnToPredict_]-self.mean_y)/self.std_y
        
        self.train_timeseries = TimeSeries.from_dataframe(new_train_set, time_column_, columnToPredict_, freq=dataTimeStep_)

        self.model = StatsForecastAutoARIMA
        
    def getModelName(self):
        return self.MODEL_NAME
    
    def gridSearch(self):
        self.run_and_fit_model()
    
    def run_and_fit_model(self):
        self.model = self.model().fit(self.train_timeseries)

    def predict(self):
        forecast = self.model.predict(self.points_to_predict)
        return (forecast._xa.values.flatten()*self.std_y+self.mean_y)
