# -*- coding: utf-8 -*-
"""

@author: Gonçalo Mateus

MODEL_NAME
----------


Model Overview
--------------
1) Ensemble
2) Based on Mixure of Experts (weights)
3) Interpretable model
4) Real-time re-trainning
5) Allows past, future, and static co-variates
----------------------------------------------

Notes
-----


"""

#%% Imports

import pandas as pd
from datetime import timedelta
from datetime import datetime
import time

#from Functions_auxFile import getTwitterData, splitFilesPath, loadData

from EAMDrift_model.Ensemble_Model_Class import EnsembleModelClass

import warnings
warnings.filterwarnings("ignore")


#%% RUN MODEL

##############################################################################
# Load files 
##############################################################################
#Load twitter data
#twitter_cleaned_df = getTwitterData(splitFilesPath('Twitter_Data/', 'cleaned')[0])
#twitter_all_df = getTwitterData(splitFilesPath('Twitter_Data/', 'cleaned')[1])

#Load data of PL layer - 7 server + 1 average
#pl_layer_6_hours = loadData("PL", "6_hours")
#pl_layer_30_minutes = loadData("PL", "30_minutes")

#Load data of BL layer - 9 server + 1 average
#bl_layer_6_hours = loadData("BL", "6_hours")
#bl_layer_30_minutes = loadData("BL", "30_minutes")

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/MarketArrivals.csv')
     
##############################################################################
# Model run 
##############################################################################
if __name__ == '__main__':  
    
    #dataframe - time e feature to predict - se tem mais retiramos
    #covariadas - past
    #covariadas - static
    #twitter - 
    #tudo isto com o mm tamanho - datas iguais - etc...
    
    # Make Trainning set
    dataframe = df.copy()
    dataframe = dataframe[["date", "quantity"]]
    
    dates = []
    previous = datetime(2020,10,29)
    for x in range(len(dataframe)):
        dates.append(previous+timedelta(hours=6))
        previous = previous+timedelta(hours=6)
    
    dataframe["date"] = dates


    start = time.time()
    models = ["ExponentialSmoothing", "ExponentialSmoothing"] #models to use
    
    
    ensemble_model = EnsembleModelClass(timeseries_df_ = dataframe,                           
                                          columnToPredict_ = "quantity",          
                                          time_column_ = "date",
                                          models_to_use_ = models,
                                          #trainning_samples_size_ = 100,  
                                          trainning_points_ = 150,               
                                          prediction_points_ = 4,                
                                          to_extract_features_ = False)    
     
    trainning_dataframe_index, trainning_dataframe, errors = ensemble_model.create_trainning_set()
     
    #test_models

    end = time.time()
    print("\nTime: " + str(round(((end - start)/60),2)) + " minutes")
        
    # Train and run ensemble method
    
    # Predict
    