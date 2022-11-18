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


# %% Imports

import pandas as pd
import math
from tqdm import tqdm
from typing import Optional
from joblib import Parallel, delayed

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

from river import drift

from EAMDrift_model.Models import ModelsDB
from EAMDrift_model.rulefit import RuleFit

from darts.timeseries import TimeSeries

import warnings
warnings.filterwarnings("ignore")

# %% EAMDriftModel

class EAMDriftModel:

    def __init__(self,
                 timeseries_df_: object,
                 columnToPredict_: str,
                 time_column_: str,
                 models_to_use_: list,
                 dataTimeStep_: str,
                 covariates_: Optional[object] = None,
                 categorical_covariates_ : Optional[list] = None,
                 covariates_ma_: Optional[int] = None,
                 error_metric_: Optional[str] = "MAPE",
                 trainning_samples_size_: Optional[int] = None,
                 trainning_points_: int = 150,
                 fit_points_size_: int = 150,
                 prediction_points_: int = 4,
                 to_extract_features_: bool = True,
                 n_jobs_: Optional[int] = None):
        """EAMDrift_model

        Parameters
        ----------
        timeseries_df_ : object
            Order .
        columnToPredict_ : str
            The .
        time_column_ : str
            The .
        models_to_use_: list
            The .
        dataTimeStep_: str
            The .
        trainning_samples_size_: Optional[int]
            If no value is given, the model will try to make the max number of 
            possibles subsets.
        trainning_points_: int
            The .
            Default value is 150.
        prediction_points_: int
            The .
            Default value is 4.
        prediction_points_: bool
            The .
            Default value is True.
        n_jobs_: Optional[int]
            The .
            Default value is 4.
        """

        self.timeseries_df = timeseries_df_
        self.columnToPredict = columnToPredict_
        self.time_column = time_column_
        self.models_to_use = models_to_use_.copy()
        self.dataTimeStep = dataTimeStep_
        self.covariates = covariates_
        self.categorical_covariates = categorical_covariates_
        self.covariates_ma = covariates_ma_
        self.error_metric = error_metric_
        self.trainning_samples_size = trainning_samples_size_
        self.trainning_points = trainning_points_
        self.fit_points_size = fit_points_size_
        self.prediction_points = prediction_points_
        self.to_extract_features = to_extract_features_
        self.to_retrain = False
        self.last_drift = timeseries_df_.date.describe().top
        self.n_jobs = n_jobs_
        self.verbose = False

        if(error_metric_ != "MAE" and error_metric_ != "MSE" and error_metric_ != "MAPE"):
            raise Exception('Error metric inserted is not implemented.')
        
        
        
        #OTHERS
        self.models_to_use_original = models_to_use_.copy()
        self.train_set = pd.DataFrame([])
        self.extract_features_data_y = []
        self.models_forecast_df = []
        self.ensemble_model = None
        #import sys, os
        #sys.stdout = open(os.devnull, 'w')
        #tqdm.disable()
        
        self.features_col = []

    def create_trainning_set(self):
        """
        Split dataframe in subsets to train the ensemble model

        Returns a dataframe to be used in trainning
        """
        
        self.models_to_use = self.models_to_use_original.copy()
        
        # Pre calculation
        # Max number of splits that can happen
        max_splits = math.floor(
            (len(self.timeseries_df)-self.trainning_points)/self.prediction_points)
        self.trainning_samples_size = max_splits if (
            self.trainning_samples_size == None) else self.trainning_samples_size

        # Exceptions
        if(self.trainning_points > len(self.timeseries_df)):
            raise Exception(
                'The number of points to train is biger than the dataframe dimension.')
        elif(max_splits < self.trainning_samples_size):
            raise Exception(
                'Not enough points to create trainning set with your requirements.')

        # Calculate space
        necessarySpace = self.trainning_points + \
            self.trainning_samples_size*self.prediction_points
        additionallySpace = len(self.timeseries_df)-necessarySpace

        # Splitting dataframe
        data, extract_features_data, extract_features_data_y, covariates_data = self.__splitting_dataframe(
            additionallySpace)
        
        #
        self.extract_features_data_y = extract_features_data_y

        # Feature Extraction
        split_dataframe = pd.DataFrame(data, columns=[
                                       'df_fixed_index', 'y_index', 'df_extended_index', 'new_points_index'])
        if(self.to_extract_features):
            extracted_additionally_features = self.__extract_statistics_info(
                extract_features_data, extract_features_data_y)
            covariates_data = pd.concat(
                [covariates_data, extracted_additionally_features], axis=1).reindex(covariates_data.index)
        
        # Adding models accuracys
        if (self.n_jobs == None):
            errors_df, models_forecast_df = self.__test_models_init(
                train_dataframe_index=split_dataframe)
        else:
            errors_df, models_forecast_df = self.__test_models_init_jobs(
                train_dataframe_index=split_dataframe)

        # Calculate the best models
        errors_df = self.__calc_best_model(errors_df)
        
        num = 0
        for x in errors_df[errors_df.columns].mean():
            print(f"{errors_df.columns[num]}: {round(x,2)}");
            num = num + 1
        
        # Save trainning set
        self.train_set = covariates_data.copy()
        self.train_set["Best Model"] = errors_df["Best Model"].values
        self.models_forecast_df = models_forecast_df.reindex(sorted(models_forecast_df.columns), axis=1)
        
        self.extract_features_data_y = pd.DataFrame(self.extract_features_data_y, columns=["feat_y"])
        new_aux = pd.DataFrame(self.train_set["Best Model"].value_counts())
        new_aux = new_aux[new_aux["Best Model"] == 1].index.values
        for x in new_aux:
            print(f"Removed {x}")
            self.models_to_use.remove(x)            
            self.models_forecast_df.drop([x], axis = 1, inplace=True)
            index_ = self.train_set[self.train_set["Best Model"] == x].index[0]
            
            self.train_set=self.train_set.drop(index=(index_))
            self.models_forecast_df=self.models_forecast_df.drop(index=(index_))
            self.extract_features_data_y=self.extract_features_data_y.drop(index=(index_))
            
        self.train_set = self.train_set.reset_index(drop=True)
        self.models_forecast_df = self.models_forecast_df.reset_index(drop=True)
        self.extract_features_data_y =  self.extract_features_data_y["feat_y"].values

        return self.train_set

    def __splitting_dataframe(self, additionally_space: int):
        """
        Splits the dataframe in several different sets

        additionally_space: int -
        """
        data = []
        extract_features_data = []
        extract_features_data_y = []
        covariates_data = [] #para fazer match com o resto do treino
        
        previousEnd = 0
        for x in tqdm(range(self.trainning_samples_size), desc="Splitting dataframe", disable=self.verbose):
            initIndex = (x * (self.prediction_points)) + additionally_space
            splitIndex = initIndex + self.trainning_points
            endIndex = splitIndex + self.prediction_points
            data.append([[initIndex, splitIndex], [splitIndex, endIndex], [
                        additionally_space, splitIndex], [previousEnd, splitIndex]])

            df_fixed = pd.DataFrame(
                self.timeseries_df.iloc[initIndex: splitIndex][self.columnToPredict]).reset_index(drop=True)
            df_fixed["id"] = x
            df_fixed["time"] = list(range(0, self.trainning_points))

            extract_features_data = df_fixed if (x == 0) else pd.concat(
                [extract_features_data, df_fixed], ignore_index=True, sort=True).reset_index(drop=True)
            extract_features_data_y.append(
                (self.timeseries_df.iloc[splitIndex: endIndex][self.columnToPredict]).mean())

            # TODO: ver mlhr
            covariates_aux = self.covariates.iloc[(splitIndex-self.covariates_ma): splitIndex]            
            covariates_split_processed = []
            
            for col in self.covariates.columns:
                if(col in self.categorical_covariates):
                    covariates_split_processed.append(covariates_aux[col].value_counts().iloc[0])
                else:
                    covariates_split_processed.append(covariates_aux[col].sum())
           
            covariates_data.append(covariates_split_processed)

            previousEnd = splitIndex

        covariates_data = pd.DataFrame(covariates_data, columns=[self.covariates.columns])  #alterar dps 

        return data, extract_features_data, extract_features_data_y, covariates_data

    def __extract_statistics_info(self, features_data: object, features_data_y: list):
        """
        Returns a dataframe with features filtered

        features_data: object -
        features_data_y: list -
        """
        from tsfresh.feature_extraction import MinimalFCParameters
        extracted_features = extract_features(features_data,
                                              column_id="id",
                                              column_sort="time",)
                                              #default_fc_parameters=MinimalFCParameters())

        impute(extracted_features)
        features_filtered = select_features(extracted_features,
                                            pd.Series(features_data_y))
        
        self.features_col = features_filtered.columns
            
        

        print(
            f"Note: {len(features_filtered.columns)} features selected from a total of {len(extracted_features.columns)}")

        return features_filtered

    def __test_models_init(self, train_dataframe_index: object):
        """
        Returns a dataframe with features filtered

        train_dataframe_index: object -
        """

        df_fixed_index = train_dataframe_index[["df_fixed_index"]]  # index x
        y_index = train_dataframe_index[["y_index"]]  # index y
        errors_df = pd.DataFrame(columns=self.models_to_use)
        models_forecast_df = pd.DataFrame(columns=self.models_to_use)

        for index in tqdm(range(len(df_fixed_index)), desc="Inserting accuracys", disable=self.verbose):

            train_index = df_fixed_index.iloc[index][0]
            val_index = y_index.iloc[index][0]

            train_set = self.timeseries_df.iloc[train_index[0]: train_index[1]]
            val_set = self.timeseries_df.iloc[val_index[0]: val_index[1]][self.columnToPredict]
            models_class = ModelsDB(train_df_=train_set, pointsToPredict_=self.prediction_points,
                                    columnToPredict_=self.columnToPredict, time_column_=self.time_column, dataTimeStep_=self.dataTimeStep)

            errors = []
            models_forecast = []
            for model_name in self.models_to_use:
                ModelClass = models_class.run_models(model_name)
                ModelClass.run_and_fit_model()
                forecast = ModelClass.predict()
                mae_error, mse_error, mape_error = self.compute_errors(
                    forecast, val_set)
                
                models_forecast.append(forecast)
                
                if(self.error_metric == "MAE"):
                    errors.append(mae_error)
                elif(self.error_metric == "MSE"):
                    errors.append(mse_error)
                else:
                    errors.append(mape_error)

            errors_df.loc[index] = errors
            models_forecast_df.loc[index] = models_forecast
            
        return errors_df, models_forecast_df

    def __test_models_init_jobs(self, train_dataframe_index: object):
        """
        Returns a dataframe with features filtered

        train_dataframe_index: object -
        """

        df_fixed_index = train_dataframe_index[["df_fixed_index"]]  # index x
        y_index = train_dataframe_index[["y_index"]]  # index y
        errors_df = pd.DataFrame(columns=self.models_to_use)
        models_forecast_df = pd.DataFrame(columns=self.models_to_use)

        for x in tqdm(range(len(self.models_to_use)), desc=f"Inserting accuracys ({len(self.models_to_use)} models)", disable=self.verbose):
            model_name = self.models_to_use[x]
            result = Parallel(n_jobs=self.n_jobs)(delayed(self.test_models_init_jobs_aux)(
                train_dataframe_index, model_name, i, y_index, df_fixed_index) for i in range(len(df_fixed_index)))
            
            result=pd.DataFrame(result, columns=["err", "forecast"])
            errors_df[model_name] = result["err"].values
            models_forecast_df[model_name] = result["forecast"].values
            
        return errors_df, models_forecast_df

    def test_models_init_jobs_aux(self, train_dataframe_index: object, model_name: str, index: int, y_index: object, df_fixed_index: object):
        """
        Returns erroc metric value

        train_dataframe_index: object -
        """

        train_index = df_fixed_index.iloc[index][0]
        val_index = y_index.iloc[index][0]

        train_set = self.timeseries_df.iloc[train_index[0]: train_index[1]]
        val_set = self.timeseries_df.iloc[val_index[0]: val_index[1]][self.columnToPredict]
        models_class = ModelsDB(train_df_=train_set, pointsToPredict_=self.prediction_points,
                                columnToPredict_=self.columnToPredict, time_column_=self.time_column, dataTimeStep_=self.dataTimeStep)

        ModelClass = models_class.run_models(model_name)
        ModelClass.run_and_fit_model()
        forecast = ModelClass.predict()

        mae_error, mse_error, mape_error = self.compute_errors(
            forecast, val_set)

        if(self.error_metric == "MAE"):
            error = mae_error
        elif(self.error_metric == "MSE"):
            error = mse_error
        else:
            error = mape_error

        return [error, forecast]

    def __calc_best_model(self, errors_dataframe: object):
        """
        calc_best_model

        errors_dataframe: object -
        """
        #print(errors_dataframe)
        errors_dataframe['Best MAPE'] = errors_dataframe[errors_dataframe.columns.values].min(
            axis='columns')

        minCol = []
        for line in tqdm(range(len(errors_dataframe)), desc="Finding best models", disable=self.verbose):
            valueToFind = errors_dataframe.iloc[line]["Best MAPE"]
            count = 0
            for x in errors_dataframe.iloc[line]:
                if(x == valueToFind):
                    minCol.append(
                        errors_dataframe.columns.values[count].split("_MAPE")[0])
                    break
                count = count + 1

        errors_dataframe['Best Model'] = minCol

        return errors_dataframe

    def __detectDrift(self, drift_type: Optional[str] = "KSWIN"):  # TODO: - TER ERROS
        """
        Alerts if any drift was detected.

        drift_type: Optional[str]
            Default value is "KSWIN".
        """

        drift_detector = None

        if(drift_type == "KSWIN"):
            drift_detector = drift.KSWIN(alpha=0.0001)
        elif(drift_type == "KSWIN"):
            drift_detector = drift.KSWIN(alpha=0.0001)
        else:
            raise Exception('Drift inserted is not implemented.')

        drifts = []
        for i, val in enumerate(self.timeseries_df):
            # Data is processed one sample at a time
            drift_detector.update(val)
            if (drift_detector.change_detected):
                # The drift detector indicates after each sample if there is a drift in the data
                #print(f'Change detected at index {i}')
                drifts.append(i)
                drift_detector.reset()

        if(self.last_drift >= drifts[-1]):
            self.last_drift = drifts[-1]
            return True

        return False

    def fitEnsemble(self):
        if(self.train_set.empty):
            raise Exception('No trainning set created')
    
        y = self.train_set["Best Model"] 
        print(self.train_set["Best Model"].value_counts())
        X = self.train_set.copy()
        X = X.drop(["Best Model"], axis=1)
        
        features = X.columns
        X = X.values
        
        y_class = y.copy()
        #N = X.shape[0]

        self.ensemble_model = RuleFit(tree_size=4, sample_fract='default', max_rules=2000,
                     memory_par=0.01, tree_generator=None,
                     rfmode='classify', lin_trim_quantile=0.025,
                     lin_standardise=True, exp_rand_tree_size=True, random_state=1) 
    
        self.ensemble_model.fit(X, y_class, feature_names=features)
        y_pred = self.ensemble_model.predict(X)
        #print(print(pd.DataFrame(y_pred).value_counts()))
        y_proba = self.ensemble_model.predict_proba(X)
        y_proba = pd.DataFrame(y_proba)
        #print(y_proba)
        insample_acc = sum(y_pred == y_class) / len(y_class)
        print(f"Ensemble Model accuracy: {insample_acc}")

        rules = self.ensemble_model.get_rules()
    
        rules = rules[rules.coef != 0].sort_values(by="support")
        #num_rules_rule = len(rules[rules.type == 'rule'])
        #num_rules_linear = len(rules[rules.type == 'linear'])
        #print(rules.sort_values('importance', ascending=False))
       
        count = 0
        for x in self.models_forecast_df.columns:
            self.models_forecast_df[x] = self.models_forecast_df[x] * y_proba[count]
            count+=1
          
        all_sum = self.models_forecast_df.sum(axis='columns')

        forecasts = []
        for x in range(len(all_sum)):
            forecasts.append(all_sum.iloc[x].mean())
        
        print(f"Trainning error: {mean_absolute_percentage_error(forecasts, self.extract_features_data_y)}")
            
        #return y_proba, self.extract_features_data_y, y_pred, rules,forecasts
        return rules

    def predict(self, n: Optional[int] = None, series: Optional[TimeSeries] = None, covariates: Optional[TimeSeries] = None):
        if (n == None):
            n = self.prediction_points
    
        # TODO: CASO AS SERIES NÃO SEJA NULL
            #CRIAR NOVO CONJUNTO DE TREINO / ADICIONAR ESTE NOVO CONJUNTO AO CONJUNTO ANTIGO
            #NAO PRECISA DE TREINAR
            #ISTO É MAIS PARA ELE CONSEGUIR PREVER NO ENSEMBLE

        # TODO: TESTAR AQUI COM FIT DE TODO O CONJUNTO DE TREINO
        df_fixed = pd.DataFrame(
            self.timeseries_df.iloc[-self.trainning_points:][self.columnToPredict]).reset_index(drop=True)
        
        df_fixed["id"] = 0
        df_fixed["time"] = list(range(0, self.trainning_points))

        extracted_features = extract_features(df_fixed,                         
                                              column_id="id",     
                                              column_sort = "time",
                                              column_value=self.columnToPredict,               
                                              disable_progressbar=True,
                                              n_jobs=1)        
        
        extracted_features = extracted_features[self.features_col]
        
        covariates_aux = self.covariates.iloc[-self.covariates_ma:]           
        covariates_split_processed = []
        
        for col in self.covariates.columns:
            if(col in self.categorical_covariates):
                covariates_split_processed.append(covariates_aux[col].value_counts().iloc[0])
            else:
                covariates_split_processed.append(covariates_aux[col].sum())
        covariates_split_processed = pd.DataFrame([covariates_split_processed], columns=[self.covariates.columns])
        
        covariates_data = pd.concat([covariates_split_processed, extracted_features], axis=1)
        y_prob = pd.DataFrame(self.ensemble_model.predict_proba(covariates_data.values))
        
        models_class = ModelsDB(train_df_=self.timeseries_df.iloc[-self.fit_points_size:], pointsToPredict_=n,
                                columnToPredict_=self.columnToPredict, time_column_=self.time_column, dataTimeStep_=self.dataTimeStep)
                
        models_forecast = []
        for model_name in self.models_to_use:
            ModelClass = models_class.run_models(model_name)
            ModelClass.run_and_fit_model()
            forecast = ModelClass.predict()            
            models_forecast.append(forecast)
            
        models_forecast = pd.DataFrame([models_forecast], columns=self.models_to_use)
        models_forecast = models_forecast.reindex(sorted(models_forecast.columns), axis=1)
        
        count = 0
        for x in models_forecast.columns:
            models_forecast[x] = models_forecast[x] * y_prob[count]
            count+=1
          
        prediction = models_forecast.sum(axis='columns')

        return prediction[0]

    def add_and_predict(self, newObservations: Optional[object] = None, newCovariates: Optional[object] = None): # TODO: c

        # TODO: check dates
        self.timeseries_df = pd.concat([self.timeseries_df, newObservations], axis=0).reset_index(drop=True)
        self.covariates = pd.concat([self.covariates, newCovariates], axis=0).reset_index(drop=True)
        
        # mm que nao haja drift se as accuracys tiverem a descer, retreina
        # adiciona os novos pontos ao dataframe de treino e ao timeseries
        """
        if(self.__detectDrift()):
            self.to_retrain = True

            if(self.to_retrain()):
                # check accuracys
                # is decreasing?
                # re train
                # self.to_retrain=False
                print("")
        """
        # Predict
        
        prediction = self.predict(self.prediction_points)
        
        return prediction

    def historical_forecasts(self): # TODO: w
    
        #ciclo do add and predict tudo aqui dentrob
    
        return ""

    def run_fit_predict(self): # TODO: w
        return ""
    
    def compute_errors(self, forecast, val):
        """
            Compute model errors

            forecast - forecast
            val - validation set
        """

        # Compute Errors (MAE, MSE and MAPE)
        mae_cross_val = mean_absolute_error(forecast, val)
        mse_cross_val = mean_squared_error(forecast, val)
        mape = mean_absolute_percentage_error(forecast, val)

        return mae_cross_val, mse_cross_val, mape

    def print_report(): # TODO: w
        from tabulate import tabulate
        data = [[1, 'Liquid', 24, 12],
        [2, 'Virtus.pro', 19, 14],
        [3, 'PSG.LGD', 15, 19],
        [4,'Team Secret', 10, 20]]
        print (tabulate(data, headers=["Pos", "Team", "Win", "Lose"]))
        print("")
