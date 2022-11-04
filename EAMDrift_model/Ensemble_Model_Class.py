# -*- coding: utf-8 -*-
"""

@author: Gonçalo Mateus

Ensemble Model Class
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
import math
from tqdm import tqdm

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

from EAMDrift_model.Models import ModelsDB

import warnings
warnings.filterwarnings("ignore")

#%% EnsembleModelClass

class EnsembleModelClass:

    def __init__(self,
                 timeseries_df_: object, 
                 columnToPredict_: str,
                 time_column_: str,
                 models_to_use_: list,
                 trainning_samples_size_: int = None,
                 trainning_points_: int = 150,
                 prediction_points_: int = 4,
                 to_extract_features_: bool = True):
  
        self.timeseries_df = timeseries_df_
        self.columnToPredict = columnToPredict_
        self.time_column = time_column_
        self.models_to_use = models_to_use_
        self.trainning_samples_size = trainning_samples_size_
        self.trainning_points = trainning_points_
        self.prediction_points = prediction_points_
        self.to_extract_features = to_extract_features_
    
    def create_trainning_set(self):
        """
            Split dataframe in subsets to train the ensemble model
            
            Returns a dataframe to be used in trainning
                    
            timeseries_df: object -
            columnToPredict: str - 
            time_column: str - 
            trainning_samples_size: int = None -
            trainning_points: int = 150 - 
            prediction_points: int = 4 - 
            extract_features: bool = True - 
        """
        
        #Pre calculation        
        max_splits = math.floor((len(self.timeseries_df)-self.trainning_points)/self.prediction_points) #Max number of splits that can happen
        self.trainning_samples_size = max_splits if (self.trainning_samples_size == None) else self.trainning_samples_size
        
        #Exceptions
        if(self.trainning_points > len(self.timeseries_df)):
            raise Exception('The number of points to train is biger than the dataframe dimension.')
        elif(max_splits < self.trainning_samples_size):
            raise Exception('Not enough points to create trainning set with your requirements.')
        
        #Calculate space
        necessarySpace = self.trainning_points + self.trainning_samples_size*self.prediction_points
        additionallySpace = len(self.timeseries_df)-necessarySpace
        
        #Splitting dataframe
        data, extract_features_data, extract_features_data_y = self.splitting_dataframe(additionallySpace)
                
        #Feature Extraction
        split_dataframe = pd.DataFrame(data, columns=['df_fixed_index', 'y_index', 'df_extended_index', 'new_points_index'])
        if(self.to_extract_features):
            extracted_additionally_features = self.extract_statistics_info(extract_features_data, extract_features_data_y)
            split_dataframe = pd.concat([split_dataframe, extracted_additionally_features], axis=1).reindex(split_dataframe.index)
        
        #Adding models accuracys
        errors_df = self.test_models_init(train_dataframe_index = split_dataframe)
        
        #Calculate the best models 
        errors_df = self.calc_best_model(errors_df)
        
        return split_dataframe, self.timeseries_df, errors_df
      
    def splitting_dataframe(self, additionally_space: int):
        """
            Splits the dataframe in several different sets
                    
            timeseries_df: object -
            columnToPredict: str - 
            trainning_samples_size: int = None -
            trainning_points: int - 
            prediction_points: int - 
            additionallySpace: int - 
        """
        data = []
        extract_features_data = []
        extract_features_data_y = []
        previousEnd = 0
        for x in tqdm(range(self.trainning_samples_size), desc="Splitting dataframe"):
            initIndex = (x * (self.prediction_points)) + additionally_space
            splitIndex = initIndex + self.trainning_points
            endIndex = splitIndex + self.prediction_points
            data.append([[initIndex, splitIndex], [splitIndex, endIndex], [additionally_space, splitIndex], [previousEnd,splitIndex]])  
            
            df_fixed = pd.DataFrame(self.timeseries_df.iloc[initIndex: splitIndex][self.columnToPredict]).reset_index(drop=True)
            df_fixed["id"] = x
            df_fixed["time"] = list(range(0, self.trainning_points))
            
            extract_features_data = df_fixed  if (x==0) else pd.concat([extract_features_data, df_fixed], ignore_index=True, sort=True).reset_index(drop=True)
            extract_features_data_y.append((self.timeseries_df.iloc[splitIndex: endIndex][self.columnToPredict]).mean())

            previousEnd = splitIndex
            
        return data, extract_features_data, extract_features_data_y

    def extract_statistics_info(self, features_data: object, features_data_y: list):
        """
            Returns a dataframe with features filtered
                    
            features_data: object -
            features_data_y: list -
        """
        extracted_features = extract_features(features_data, 
                                              column_id="id", 
                                              column_sort="time")
        impute(extracted_features)    
        features_filtered = select_features(extracted_features, 
                                            pd.Series(features_data_y))
        
        print(f"Note: {len(features_filtered.columns)} features selected from a total of {len(extracted_features.columns)}")
        
        return features_filtered

    def test_models_init(self, train_dataframe_index: object):

        df_fixed_index = train_dataframe_index[["df_fixed_index"]] #index x
        y_index = train_dataframe_index[["y_index"]] #index y 
        errors_df = pd.DataFrame(columns = self.models_to_use)
        
        for index in tqdm(range(len(df_fixed_index)), desc="Inserting accuracys"):
            
            train_index = df_fixed_index.iloc[index][0]
            val_index = y_index.iloc[index][0]
            
            train_set = self.timeseries_df.iloc[train_index[0]: train_index[1]]
            val_set = self.timeseries_df.iloc[val_index[0]: val_index[1]][self.columnToPredict]
            models_class = ModelsDB(train_df_=train_set, pointsToPredict_ = self.prediction_points, columnToPredict_ = self.columnToPredict, time_column_ = self.time_column)

            errors = []
            for model_name in self.models_to_use:
                ExponentialSmoothingClass = models_class.run_models(model_name)
                ExponentialSmoothingClass.run_and_fit_model(28)
                forecast = ExponentialSmoothingClass.predict()    
                
                mae_error, mse_error, mape_error = self.compute_errors(forecast, val_set)            
                errors.append(mape_error)
                            
            errors_df.loc[index] = errors

        return errors_df

    def calc_best_model(self, errors_dataframe: object): 
        """
            calc_best_model
                    
            errors_dataframe: object -
        """
        
        errors_dataframe['Best MAPE'] = errors_dataframe[errors_dataframe.columns.values].min(axis='columns') 
        
        minCol = []
        for line in tqdm(range(len(errors_dataframe)), desc="Finding best models"):
            valueToFind = errors_dataframe.iloc[line]["Best MAPE"]
            count = 0 
            for x in errors_dataframe.iloc[line]:
                if(x == valueToFind):
                    minCol.append(errors_dataframe.columns.values[count].split("_MAPE")[0])
                    break;
                count = count + 1
                
        errors_dataframe['Best Model'] = minCol 
        
        return errors_dataframe

    def detectDrift(self): #FAZER
        return ""

    def fitEnsemble(self): #FAZER
        return ""

    def predict(self): #FAZER
        return ""

    def historical_forecasts(self): #FAZER
        return ""

    def compute_errors(self, forecast, val):
        """
            Compute model errors
                
            forecast - forecast
            val - validation set
        """
                
        #Compute Errors (MAE, MSE and MAPE)
        mae_cross_val = mean_absolute_error(forecast, val)
        mse_cross_val = mean_squared_error(forecast, val)
        mape = mean_absolute_percentage_error(forecast, val)
            
        return mae_cross_val, mse_cross_val, mape

