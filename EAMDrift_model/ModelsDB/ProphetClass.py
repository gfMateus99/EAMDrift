# -*- coding: utf-8 -*-
"""
ProphetClass
"""

from EAMDrift_model.ModelsDB.Models_Interface import ModelsInterface

from darts import TimeSeries
from prophet import Prophet
import pandas as pd

import logging
cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.disabled = True

class ProphetClass(ModelsInterface):
    
    def __init__(self, modelName_: str, train_df_: object, pointsToPredict_: int, columnToPredict_: str, time_column_: str, dataTimeStep_: str):
        self.MODEL_NAME = modelName_
        self.points_to_predict = pointsToPredict_
        self.train_df = train_df_.copy()
        self.train_timeseries = TimeSeries.from_dataframe(train_df_, time_column_, columnToPredict_, freq=dataTimeStep_)
        self.dataTimeStep = dataTimeStep_
        
        self.model = Prophet
        
        self.train_df = self.train_df[["date", columnToPredict_]]
        self.train_df.columns = ['ds', 'y']
        
        self.mean_y = self.train_df["y"].mean()
        self.std_y = self.train_df["y"].std()
        self.train_df["y"] = (self.train_df["y"]-self.mean_y)/self.std_y
        
    def getModelName(self):
        return self.MODEL_NAME
    
    def gridSearch(self):
        self.run_and_fit_model()
    
    def run_and_fit_model(self):
        
        self.model = self.model(yearly_seasonality = True,
                        daily_seasonality = True,
                        weekly_seasonality = True,
                        growth = 'linear',
                        seasonality_mode = "multiplicative",
                       )

        self.model.fit(self.train_df)

    def predict(self):
        future = self.model.make_future_dataframe(periods=self.points_to_predict, freq=self.dataTimeStep)
        forecast = self.model.predict(future)

        y_pred = forecast['yhat'].values*self.std_y+self.mean_y
        y_pred = pd.DataFrame(y_pred)
        y_pred = y_pred[-self.points_to_predict:][0].values
        
        return y_pred








