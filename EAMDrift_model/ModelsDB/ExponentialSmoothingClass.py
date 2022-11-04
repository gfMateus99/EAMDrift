# -*- coding: utf-8 -*-
"""
ExponentialSmoothingClass
"""

class ExponentialSmoothingClass:
    
    def __init__(self, modelName_: str, train_df_: object, pointsToPredict_: int, columnToPredict_: str, time_column_: str):
        from darts import TimeSeries
        from darts.models import ExponentialSmoothing
        
        self.MODEL_NAME = modelName_
        self.points_to_predict = pointsToPredict_
        #self.train_df = train_df_
        self.train_timeseries = TimeSeries.from_dataframe(train_df_, time_column_, columnToPredict_, freq='6H')
        self.model = ExponentialSmoothing
        
    def getModelName(self):
        return self.MODEL_NAME
    
    def gridSearch(self):
       
        param = {
                  "seasonal_periods_val": [4,8,28]
                }
        
        self.model = self.model.gridsearch(param, self.train_timeseries, use_fitted_values=True)
        self.model.fit(self.train_timeseries)
    
    def run_and_fit_model(self, seasonal_periods_val_: int):
        self.model = self.model(seasonal_periods=seasonal_periods_val_).fit(self.train_timeseries)

    def predict(self):
        forecast = self.model.predict(self.points_to_predict)
        return forecast._xa.values.flatten()
