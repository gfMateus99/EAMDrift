# -*- coding: utf-8 -*-
"""
ExponentialSmoothingClass
"""

from EAMDrift_model.ModelsDB.Models_Interface import ModelsInterface

import pandas as pd

from darts import TimeSeries
import statsmodels.api as stm

from itertools import product

class SARIMAXClass(ModelsInterface):
    
    def __init__(self, modelName_: str, train_df_: object, pointsToPredict_: int, columnToPredict_: str, time_column_: str, dataTimeStep_: str):
        self.MODEL_NAME = modelName_
        self.points_to_predict = pointsToPredict_
        self.train_df = train_df_
        
        new_train_set = train_df_.copy()
        self.train_timeseries = TimeSeries.from_dataframe(new_train_set, time_column_, columnToPredict_, freq=dataTimeStep_)

        self.mean_y = self.train_timeseries._xa.values.flatten().mean()
        self.std_y = self.train_timeseries._xa.values.flatten().std()
        self.train_timeseries = self.train_timeseries._xa.values.flatten()
        self.train_timeseries = (self.train_timeseries-self.mean_y)/self.std_y
                
        self.model = stm.tsa.statespace.SARIMAX
        
        
    def get_best_model(self, parameters_list):
        """
            Return dataframe with parameters, corresponding AIC and BIC
            
            parameters_list - list with (p, d, q, P, D, Q) tuples
            s - length of season
            train - the train variable
        """
        results = []
        
        for param in parameters_list:
            try: 
                model = stm.tsa.statespace.SARIMAX(self.train_timeseries, order=(param[0], param[1], param[2]), seasonal_order=(param[3], param[4], param[5], param[6])).fit(disp=-1)
            except:
                continue   
            aic = model.aic
            bic = model.bic
            results.append([param, aic, bic])
        
        result_df = pd.DataFrame(results)
        result_df.columns = ['(p, d, q)x(P, D, Q)', 'AIC', 'BIC']
        
        #Sort in ascending order, lower AIC is better
        result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
        
        return result_df

    def run_sarima_model(self, p, d, q, P, D, Q, s):
        """
            Return Sarimax model with gice definitions and fit 
            
            prophet_dataframe - dataframe to train
            yearly_seasonality - yearly seasonality
            daily_seasonality - daily seasonality
            weekly_seasonality - weekly seasonality
            growth - growth of model
            seasonality_mode - mode of seasonality
            holidays - holidays to model
        """
        #Define the model
        self.model = stm.tsa.statespace.SARIMAX(self.train_timeseries, order=(p, d, q), seasonal_order=(P, D, Q, s)).fit(disp=-1)
        #self.model.fit(self.train_timeseries)
    
    def getModelName(self):
        return self.MODEL_NAME
    
    def gridSearch(self):
        self.run_and_fit_model()
    
    def run_and_fit_model(self):
        # 5: Define parameters
        p = [0,1,2,3,4]
        d = [0]
        q = [1,2,3]
        
        P = [0,1,2,3,4]
        D = [0]
        Q = [1,2,3]
        
        s = [4, 28] 
        # 6 hours * 4 = 24 hours - daily seasonality (4)
        # 7 * 6 hours * 4 = 7 days - weekly seasonality (28)
        parameters = product(p, d, q, P, D, Q, s)
        #parameters_list = list(parameters)
        
        parameters_list= [(3, 0, 3, 1, 0, 3, 28),
                            (4, 0, 3, 4, 0, 1, 28),
                            (4, 0, 3, 2, 0, 2, 28),
                            (1, 0, 2, 2, 0, 3, 28),
                            (1, 0, 2, 3, 0, 2, 28),
                            (4, 0, 3, 2, 0, 1, 28)
                            ]  
        
        result_sarima = self.get_best_model(parameters_list)
        parameters = result_sarima.iloc[0]["(p, d, q)x(P, D, Q)"]

        self.run_sarima_model(int(parameters[0]), int(parameters[1]), int(parameters[2]), int(parameters[3]), int(parameters[4]), int(parameters[5]), int(parameters[6]))

    def predict(self):
        #forecast = self.model.predict(self.points_to_predict)
        forecast = self.model.predict(start=self.train_timeseries.shape[0], end=self.train_timeseries.shape[0]+(self.points_to_predict-1))
        #print(forecast)
        return (forecast*self.std_y+self.mean_y)








