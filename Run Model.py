# -*- coding: utf-8 -*-
"""

@author: Gonçalo Mateus

EAMDrift with Google Cluster Trace
----------

"""

# %% Imports

##############################################################################
# Imports
##############################################################################

#import datetime

#from datetime import datetime
from EAMDrift_model import EAMDriftModel
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# %% Load files

##############################################################################
# Load files
##############################################################################

# Load data
data = pd.read_csv(r'data_hours_organized_reduced.csv')
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S')

# Prepare covariates 
covariates_aux = data.copy()
covariates_col = covariates_aux.columns
covariates_aux = covariates_aux.reset_index()
covariates_aux.drop(covariates_col, axis = 1, inplace=True)

# %% RUN MODEL

##############################################################################
# Model run
##############################################################################

if __name__ == '__main__':

    # Data    
    dataframe = data[["date", "cpu_usage"]].copy()    
    
    # Models to use
    models = ["TRANSFORMER", "LSTM", "LSTM2", "SARIMAX", "ExponentialSmoothing", "Prophet"]

    # Standartize
    mean_y = dataframe["cpu_usage"].mean()
    std_y = dataframe["cpu_usage"].std()
    dataframe["cpu_usage"] = (dataframe["cpu_usage"]-mean_y)/std_y

    index = 223
    points_to_predict = 6
    ensemble_model = EAMDriftModel(timeseries_df_=dataframe[0:index],
                                   columnToPredict_="cpu_usage",
                                   time_column_="date",
                                   models_to_use_=models,
                                   dataTimeStep_="1H",
                                   covariates_=covariates_aux[0:index],
                                   #categorical_covariates_ = [],
                                   covariates_ma_=7, 
                                   error_metric_="MAPE",
                                   #trainning_samples_size_ = 100,
                                   trainning_points_=100, 
                                   fit_points_size_=100,
                                   prediction_points_=points_to_predict,
                                   to_extract_features_=True,
                                   use_dates_=True,
                                   #selectBest_=1, # None selects all
                                   to_retrain_=True,
                                   n_jobs_=6
                                   )
    
    # Make Trainning set
    trainning_set_init, self_ = ensemble_model.create_trainning_set()     
    
    # Train ensemble method
    rules = ensemble_model.fitEnsemble() 
    
    # Predict
    #prediction = ensemble_model.predict() # Predict the next points_to_predict points
    
    # Predict with historical forecasts
    forecast, dates, y_true = ensemble_model.historical_forecasts(dataframe[index:], 
                                                                  forecast_horizon = int((len(dataframe)-index)/points_to_predict), 
                                                                  covariates = covariates_aux[index:])

    # Print Report
    #ensemble_model.print_report(self)
 
    # Compute errors
    print(ensemble_model.compute_errors(forecast[:-points_to_predict]*mean_y+std_y, y_true*mean_y+std_y))
       