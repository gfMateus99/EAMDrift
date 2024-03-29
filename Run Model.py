# -*- coding: utf-8 -*-
"""

@author: Anonymous


Model Overview
--------------
1) Ensemble
2) Based on Mixure of Experts (weights)
3) Interpretable model
4) Real-time re-trainning
5) Allows past, future, and static co-variates
----------------------------------------------


The dataset used in this test is the Electric Power Consumption (EPC) [1]. It measures the electric power usage in different houses in the zone of Paris, France, and for this test, just one house was chosen. The data has a 1-minute step for nearly four years but was aggregated in hours, giving us 35063 entries. As electric consumption can be related to weather, we used data from AWOS sensors available in [2] to be used as covariates.

[1] Hebrail, Georges, Berard, Alice. (2012). Individual household electric power consumption. UCI Machine Learning Repository. Accessed: 10/02/2023 [Online]. Available: doi.org/10.24432/C58K54
[2] Iowa State University. AWOS Global METAR. Accessed: 05/03/2023 [Online]. Available: mesonet.agron.iastate.edu/request/download.phtml


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
data = pd.read_csv(r'Data/LD2011_2014(MT_330).csv')
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S')
data = data.drop(["Unnamed: 0"], axis=1)

# Prepare covariates 

covariates_aux = data.copy()
covariates_col = covariates_aux.columns
covariates_aux.drop(["date", "MT_330"], axis = 1, inplace=True)

# %% RUN MODEL

##############################################################################
# Model run
##############################################################################

if __name__ == '__main__':

    # Data    
    dataframe = data[["date", "MT_330"]].copy()    
    
    # Models to use
    models = ["TRANSFORMER", "LSTM", "LSTM2", "SARIMAX", "ExponentialSmoothing", "Prophet"]

    # Standartize
    mean_y = dataframe["MT_330"].mean()
    std_y = dataframe["MT_330"].std()
    #dataframe["MT_330"] = (dataframe["MT_330"]-mean_y)/std_y

    dataframe["MT_330"] = (dataframe["MT_330"]-dataframe["MT_330"].min())/(dataframe["MT_330"].max()-dataframe["MT_330"].min())


    index = 1500
    points_to_predict = 12
    ensemble_model = EAMDriftModel(timeseries_df_=dataframe[0:index],
                                   columnToPredict_="MT_330",
                                   time_column_="date",
                                   models_to_use_=models,
                                   dataTimeStep_="1H",
                                   covariates_=covariates_aux[0:index],
                                   #categorical_covariates_ = [],
                                   covariates_ma_=7, 
                                   error_metric_="MAPE",
                                   #trainning_samples_size_ = 100,
                                   trainning_points_=150, 
                                   fit_points_size_=150,
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