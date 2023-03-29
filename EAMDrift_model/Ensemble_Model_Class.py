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

"""

# %% Imports

import pandas as pd
import math
from tqdm import tqdm
from typing import Optional
#from joblib import Parallel, delayed
import numpy as np 
from functools import reduce   

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

from tsfresh import extract_features
#from tsfresh import select_features
#from tsfresh.utilities.dataframe_functions import impute

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
                 categorical_covariates_ : Optional[list] = [],
                 covariates_ma_: Optional[int] = None,
                 error_metric_: Optional[str] = "MAPE",
                 trainning_samples_size_: Optional[int] = None,
                 trainning_points_: int = 150,
                 fit_points_size_: int = 150,
                 prediction_points_: int = 4,
                 to_extract_features_: bool = True,
                 use_dates_: bool = True,
                 selectBest_: int = None,
                 to_retrain_: Optional[bool] = True,
                 n_jobs_: Optional[int] = None):
        """EAMDrift_model

        Parameters
        ----------
        timeseries_df_ : object
            Dataframe with timeseries data.
        columnToPredict_ : str
            Column in timeseries_df_ to predict.
        time_column_ : str
            Column in timeseries_df_ with date time.
        models_to_use_: list
            List of models to use (at least two models is necessary).
        dataTimeStep_: str
            Time step of time_column_ (i.e. "6H", "1H", ...).
        covariates_: Optional[object]
            Dataframe with covariates data..
        categorical_covariates_ : Optional[list]
            Categorical covariates in covariates_ dataframe.
        covariates_ma_: Optional[int]
            Observation window for covariates at each prediction (an observation 
            of 7, corresponds to use the last 7 covariates lines).
        error_metric_: Optional[str]
            Metric to select the best models.
            Default is "MAPE"
        trainning_samples_size_: Optional[int]
            Number of samples in trainningset. If no value is given, the model 
            will try to make the max number of possibles subsets.
        trainning_points_: int
            Observation window for target varaible at each prediction (an observation 
            of 7, corresponds to use the last 7 trainning set lines).
            Default value is 150.
        fit_points_size_: int
            Number of points that the last models will use to predict. 
            Default value is 150.
        prediction_points_: int
            Number of points to predict.
            Default value is 4.
        to_extract_features_: bool
            Extract features from dataset. In case of False, the model will just 
            use covariates to predict. In case of no existence of caovariates, 
            an error will occur.
            Default value is True.
        use_dates_: bool
            To use dates covariates to predict.
            Default value is True.
        selectBest_: int
            Number of models to use to predict points. If None, all models of 
            models_to_use_ will be used.
            Default value is None.
        to_retrain_: Optional[bool]
            To retrain automatically.
            Default value is True.
        n_jobs_: Optional[int]
            Number of jobs to use.
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
        self.to_retrain = to_retrain_
        self.last_drift = timeseries_df_.date.describe().top
        self.use_dates = use_dates_
        self.selectBest = selectBest_
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
        self.scaler = None
        self.useDifferent=False
        self.new_training_set = pd.DataFrame([])
        self.forecasts_models_new = pd.DataFrame([])
        self.forecasts_models_new_dates = []
        self.rules = []
        self.retrain_periods = []
        self.rules_vec =[]
        self.proba = []
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
        if (self.fit_points_size > self.trainning_points):
            self.useDifferent=True
            max_splits = math.floor(
                ((len(self.timeseries_df)-self.fit_points_size)-self.trainning_points)/self.prediction_points)
        else: 
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
          
        
        #num = 0
        #for x in errors_df[errors_df.columns].mean():
        #    print(f"{errors_df.columns[num]}: {round(x,2)}");
        #    num = num + 1
        
        # Save trainning set
        self.train_set = covariates_data.copy()
        self.train_set["Best Model"] = errors_df["Best Model"].values
        self.models_forecast_df = models_forecast_df.reindex(sorted(models_forecast_df.columns), axis=1)
        
        self.extract_features_data_y = pd.DataFrame(self.extract_features_data_y, columns=["feat_y"])
        new_aux = pd.DataFrame(self.train_set["Best Model"].value_counts())
        new_aux = new_aux[new_aux["Best Model"] == 1].index.values
        
        
        models_to_use_aux = self.models_to_use.copy()
        for x in models_to_use_aux:
            if(x not in self.train_set["Best Model"].value_counts().index.values):
                print(f"Removed {x}")
                self.models_to_use.remove(x)   
                self.models_forecast_df.drop([x], axis = 1, inplace=True)

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

        # Reduce Features
        toDrop, not_toDrop = self.__reduce_features(self.train_set, 0.8)
                
        self.features_col = not_toDrop
        self.train_set = self.train_set.drop(toDrop, axis=1)
        
        return self.train_set, self

    def __splitting_dataframe(self, additionally_space: int):
        """
        Splits the dataframe in several different sets

        additionally_space: int -
        """
        data = []
        extract_features_data = []
        extract_features_data_y = []
        covariates_data = [] #para fazer match com o resto do treino
        
        dates = []
        
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

            last_day = self.timeseries_df.iloc[splitIndex][self.time_column]
            dates.append([last_day.day, last_day.month, last_day.year])
            
            covariates_aux = self.covariates.iloc[(splitIndex-self.covariates_ma): splitIndex]            
            covariates_split_processed = []
            
            for col in self.covariates.columns:
                if(col in self.categorical_covariates):
                    covariates_split_processed.append(covariates_aux[col].value_counts().iloc[0])
                else:
                    covariates_split_processed.append(covariates_aux[col].sum())
           
            covariates_data.append(covariates_split_processed)

            previousEnd = splitIndex

        covariates_data = pd.DataFrame(covariates_data, columns=[self.covariates.columns])  

        if(self.use_dates):
            dates = pd.DataFrame(dates, columns=["day", "month", "year"])
            covariates_data = pd.concat([covariates_data, dates], axis=1).reindex(covariates_data.index)

        return data, extract_features_data, extract_features_data_y, covariates_data

    def __extract_statistics_info(self, features_data: object, features_data_y: list):
        """
        Returns a dataframe with features filtered

        """
        extracted_features = extract_features(features_data,
                                              column_id="id",
                                              column_sort="time")
        
        #Change names
        col = []
        for x in extracted_features.columns:
            if(self.columnToPredict + "__" in x):
                col.append(x.replace(self.columnToPredict + "__", ""))
            else:
                col.append(x)
                
        extracted_features.columns = col  
                    
        df = self.__filter_features(extracted_features, features_data_y)

        print(
            f"Note: {len(df.columns)} features selected from a total of {len(extracted_features.columns)}")

        return df

    def __filter_features(self, features_data: object, features_data_y: object):
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import f_classif #ANOVA Test
                
        df = features_data.copy()
        #print(df.shape)
        # Remove columns with more than 50% of null values
        df_aux = pd.DataFrame(df.isna().sum())
        df_aux = df_aux[df_aux[0]>len(df)/2]
        df = df.drop(df_aux.index.values, axis=1)
        #print(df.shape)      
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Equal and Similar features 95% similarity
        var_threshold = VarianceThreshold(threshold=0.05)
        var_threshold.fit(df)
        features = var_threshold.transform(df)
        df = pd.DataFrame(features, columns=var_threshold.get_feature_names_out())
        #print(df.shape)
        
        # Fill the remmaing nan with 0
        df = df.fillna(0)
    
        # Correlated features - remove correlation greater than 0.95
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        df = df.drop(to_drop, axis=1)
        #print(df.shape)
        
        # ANOVA Test
        anova = SelectKBest(f_classif, k='all').fit(df, features_data_y)
        col_aux = df.columns
        to_drop=[]
        for i in range(len(anova.scores_)):
            if(anova.scores_[i] < 0.05):
                to_drop.append(col_aux[i]) 
                #print('Feature %d: %f' % (i, anova.scores_[i]))D
        df = df.drop(to_drop, axis=1)
        #print(df.shape)
    
        # Save columns
        self.features_col = df.columns

        return df   
    
    #Delete method
    def __add_features(self, new_obs: object):

        x = new_obs.copy()
        
        x.replace([np.inf, -np.inf], np.nan, inplace=True)
        x = x.fillna(0)
        
        x = self.scaler.fit_transform(x) 
        
        # Add new point to pca
        new_aux = self.pca.transform(x)
        
        column = []
        for x_ in range(new_aux.shape[1]):
            column.append("PC"+str(x_))
            
        principal_df = pd.DataFrame(data = new_aux, columns=column)
        return principal_df        
   
    def __reduce_features(self, train_set: object, l1_ratio_: int):
        test_df_train = train_set.copy()
        
        test_set = test_df_train[['Best Model']].copy()
        test_set.columns = ["y_pred"]
        test_set = test_set["y_pred"]
        
        test_df_train = test_df_train[self.features_col]
        
        analyze = ["coefficient",
                   "quantile",
                   "agg_linear_trend",
                   "matrix_profile__feature",
                   "time_reversal_asymmetry",
                   "augmented_dickey_fuller",
                   "spkt_welch_density",
                   "fourier_entropy",
                   "symmetry_looking",
                   "fft",
                   "cwt",
                   "large_standard_deviation",
                   "linear_trend__attr",
                   "permutation_entropy",
                   "mean_n_absolute_max",
                   "max_langevin_fixed_point",
                   "cid_ce"
                   ]
    
        add = []
        maintain = []
        for x in test_df_train.columns:
            found = True
            for word in analyze:
                if(word in x):
                    add.append(x)
                    found = False
                    break;
            
            if(found):
                maintain.append(x)
    
        test_df_train = test_df_train[add]
                
        test_df_train["y_pred"] = test_set
        
        #num_splits = 2
        #if(int(len(test_df_train)/100) > 1):
        #    num_splits = 2
        #else:
        #    num_splits = 1
    
        #print(num_splits)
        new_df_split = np.array_split(test_df_train, 1)
        mantain_v2=[]
        
        for x in range(len(new_df_split)):
            
            df_train_split = new_df_split[x].copy()
            df_train_split = df_train_split.reset_index(drop=True)
            
            df_test_split = df_train_split["y_pred"]
            df_train_split = df_train_split.drop(["y_pred"], axis=1)
            
            #minAlpha = 0.01
            
            #print("--------------------------")
            #print(l1_ratio_)

            features = df_train_split.columns
            pipeline = Pipeline([
                                 ('scaler',StandardScaler()),
                                 ('model',LogisticRegressionCV(l1_ratios=[l1_ratio_],
                                     cv=5, penalty='elasticnet', max_iter=100,
                                     solver='saga'))
            ])
            
            search = pipeline[1]
    
            search.fit(df_train_split,df_test_split)
            coefficients = search.coef_[0]
            importance = np.abs(coefficients)
        
            #print(len(np.array(features)[importance > 0]))
            #print(len(np.array(features)[importance == 0]))
            mantain_v2.append(np.array(features)[importance > 0])   
            
        mantain_v2 = reduce(np.intersect1d, (mantain_v2))
        
        maintain_final = np.concatenate((mantain_v2, maintain), axis=0, out=None)
        
        toDrop = []
        not_toDrop = []
        
        for z in self.features_col:
            if (z in maintain_final):
                not_toDrop.append(z)
            else:
                toDrop.append(z)
        #print(toDrop)
        
        return toDrop, not_toDrop
         
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
        from joblib import Parallel, delayed

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

        if(self.useDifferent):
            train_set = self.timeseries_df.iloc[(train_index[1]-self.fit_points_size): train_index[1]]
        else:
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
        for line in tqdm(range(len(errors_dataframe)), desc="Finding best models", disable=True):
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

    def __detectDrift(self, drift_type: Optional[str] = "KSWIN"):
        """
        Alerts if any drift was detected.

        drift_type: Optional[str]
            Default value is "KSWIN".
        """

        drift_detector = None

        if(drift_type == "KSWIN"):
            drift_detector = drift.KSWIN(alpha=0.0001)
        elif(drift_type == "ADWIN"):
            drift_detector = drift.ADWIN(delta=0.002)
        else:
            raise Exception('Drift inserted is not implemented.')
        
        dates_index = self.timeseries_df[self.time_column]
        
        drifts = []
        for i, val in enumerate(self.timeseries_df[self.columnToPredict]):
            # Data is processed one sample at a time
            drift_detector.update(val)
            if (drift_detector.change_detected):
                # The drift detector indicates after each sample if there is a drift in the data
                #print(f'Change detected at index {dates_index[i]}')
                drifts.append(dates_index[i])
                drift_detector.reset()

        if(len(drifts) > 0):
            if(self.last_drift < drifts[-1]):
                self.last_drift = drifts[-1]
                return True

        return False

    def fitEnsemble(self):
        if(self.train_set.empty):
            raise Exception('No trainning set created')
    
        train_rlefit_set = self.train_set.copy()
        #train_rlefit_set = train_rlefit_set[-500:]
    
        y = train_rlefit_set["Best Model"] 
        
        X = train_rlefit_set.copy()
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
        self.rules_vec.append(self.ensemble_model.get_rules())
        
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
        
        #print(f"Trainning error: {mean_absolute_percentage_error(forecasts, self.extract_features_data_y)}")
            
        #return y_proba, self.extract_features_data_y, y_pred, rules,forecasts
        self.rules = rules
        return rules
    
    # Fit ensemble with non-interpretable methods to test
    def fitEnsemble_aux(self):
        if(self.train_set.empty):
            raise Exception('No trainning set created')
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        train_rlefit_set = self.train_set.copy()
        #train_rlefit_set = train_rlefit_set[-500:]
    
        y = train_rlefit_set["Best Model"] 
        
        X = train_rlefit_set.copy()
        X = X.drop(["Best Model"], axis=1)
        
        #features = X.columns
        X = X.values
        
        y_class = y.copy()
        #N = X.shape[0]

        self.ensemble_model = RandomForestClassifier(max_depth=5, random_state=0)
        self.ensemble_model = SVC(gamma='auto', probability=True)

        self.ensemble_model.fit(X, y_class)
        y_pred = self.ensemble_model.predict(X)
        #print(print(pd.DataFrame(y_pred).value_counts()))
        y_proba = self.ensemble_model.predict_proba(X)
        y_proba = pd.DataFrame(y_proba)
        #print(y_proba)
        insample_acc = sum(y_pred == y_class) / len(y_class)
        print(f"Ensemble Model accuracy: {insample_acc}")

        #rules = self.ensemble_model.get_rules()
        #self.rules_vec.append(self.ensemble_model.get_rules())
        
        #rules = rules[rules.coef != 0].sort_values(by="support")
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
        
        #print(f"Trainning error: {mean_absolute_percentage_error(forecasts, self.extract_features_data_y)}")
            
        #return y_proba, self.extract_features_data_y, y_pred, rules,forecasts
        #self.rules = rules
        return "rules"
    
    def change_timeseries(self, series: object, covariates: Optional[object] = None): 
        if(covariates != None):
            self.covariates = covariates
            self.timeseries_df = series
        else:
            self.timeseries_df = series 

    def predict(self, n: Optional[int] = None, series: Optional[object] = None, covariates: Optional[object] = None):
        if (n == None):
            n = self.prediction_points
            
        if (series != None):
            if(covariates != None):
                self.change_timeseries(series, covariates) 
            else:
                self.change_timeseries(series) 

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
        
        #Change names
        col = []
        for x in extracted_features.columns:
            if(self.columnToPredict + "__" in x):
                col.append(x.replace(self.columnToPredict + "__", ""))
            else:
                col.append(x)
                
        extracted_features.columns = col  
        
        extracted_features = extracted_features[self.features_col]
        #extracted_features = self.__add_features(extracted_features)
        
        covariates_aux = self.covariates.iloc[-self.covariates_ma:]           
        covariates_split_processed = []
        
        for col in self.covariates.columns:
            if(col in self.categorical_covariates):
                covariates_split_processed.append(covariates_aux[col].value_counts().iloc[0])
            else:
                covariates_split_processed.append(covariates_aux[col].sum())
        covariates_split_processed = pd.DataFrame([covariates_split_processed], columns=[self.covariates.columns])
        
        
        dates = []
        last_day = self.timeseries_df.iloc[len(self.timeseries_df)-1][self.time_column]
        dates.append([last_day.day, last_day.month, last_day.year])
        
        if(self.use_dates):
            dates = pd.DataFrame(dates, columns=["day", "month", "year"])
            covariates_split_processed = pd.concat([covariates_split_processed, dates], axis=1).reindex(covariates_split_processed.index)
        
        covariates_data = pd.concat([covariates_split_processed, extracted_features], axis=1)
        
        covariates_data=covariates_data.fillna(0)
        covariates_data.replace([np.inf, -np.inf], 0, inplace=True)
        
        if(self.new_training_set.empty):
            self.new_training_set = covariates_data
        else:
            self.new_training_set = self.new_training_set.append(covariates_data)
        self.new_training_set=self.new_training_set.reset_index(drop=True)
        
        y_prob = pd.DataFrame(self.ensemble_model.predict_proba(covariates_data.values))
        
        self.proba.append(y_prob)
        
        if(self.selectBest != None):
            aux_df = y_prob.copy().transpose()
            aux_df["index_val"] = range(len(aux_df))
            aux_df = aux_df.sort_values(by=[0])

            toDelete = aux_df["index_val"].values[:(len(self.models_to_use)-self.selectBest)]

            for x in toDelete:
                y_prob[x] = 0
                
            total = sum(y_prob.iloc[0])

            for x in y_prob:
                y_prob[x] = y_prob[x]*1/total
                
        
        models_class = ModelsDB(train_df_=self.timeseries_df.iloc[-self.fit_points_size:], pointsToPredict_=n,
                                columnToPredict_=self.columnToPredict, time_column_=self.time_column, dataTimeStep_=self.dataTimeStep)

        models_forecast = []
        for model_name in self.models_to_use:
            #print(model_name)
            ModelClass = models_class.run_models(model_name)
            ModelClass.run_and_fit_model()
            forecast = ModelClass.predict()            
            models_forecast.append(forecast)
            
        models_forecast = pd.DataFrame([models_forecast], columns=self.models_to_use)
        
        if(self.forecasts_models_new.empty):
            self.forecasts_models_new = models_forecast
        else:
            self.forecasts_models_new = self.forecasts_models_new.append(models_forecast)
        self.forecasts_models_new=self.forecasts_models_new.reset_index(drop=True)
   
        models_forecast = models_forecast.reindex(sorted(models_forecast.columns), axis=1)
     
        new_day = 0
        import datetime
        for xindex_dates in range(n):
            
            if (new_day == 0):
                new_day = last_day + datetime.timedelta(hours=6)
            else:
                new_day = new_day + datetime.timedelta(hours=6)  
            
            if(len(self.forecasts_models_new_dates) == 0):
                self.forecasts_models_new_dates = [new_day]
            else:
                self.forecasts_models_new_dates.append(new_day)
            
        count = 0
        for x in models_forecast.columns:
            models_forecast[x] = models_forecast[x] * y_prob[count]
            count+=1
          
        prediction = models_forecast.sum(axis='columns')

        return prediction[0]

    def add_and_predict(self, newObservations: Optional[object] = None, newCovariates: Optional[object] = None): 

        self.timeseries_df = pd.concat([self.timeseries_df, newObservations], axis=0).reset_index(drop=True)
        self.covariates = pd.concat([self.covariates, newCovariates], axis=0).reset_index(drop=True)
                
        if(self.to_retrain):
            if(self.__detectDrift()):
                self.force_retrain()
        
        prediction = self.predict(self.prediction_points)
        
        return prediction

    def historical_forecasts(self, series: object, forecast_horizon: int, covariates: Optional[object] = None, retrain: Optional[bool] = True): 
    
        if (retrain):
            self.to_retrain = True
        else:
            self.to_retrain = False
        
        forecast_horizon_final = forecast_horizon*self.prediction_points

        forecast = []
        y_true = []
        dates = []
            
        first_prediction = self.predict()
        forecast = np.concatenate((forecast, first_prediction))
          
        for x in tqdm(range(0, forecast_horizon_final, self.prediction_points)):
            newObservations_historical = series[x:x+self.prediction_points]
            newCovariates_historical = np.ones(self.prediction_points)
            
            if(len(self.covariates) != 0):
                newCovariates_historical = covariates[x:x+self.prediction_points]
            
            prediction = self.add_and_predict(newObservations_historical, newCovariates_historical)
            
            forecast = np.concatenate((forecast, prediction))
            
            if(len(dates) == 0):
                dates = series[x:x+self.prediction_points]["date"].values
            else:
                dates = np.concatenate((dates, series[x:x+self.prediction_points]["date"].values))
                
            if(len(y_true) == 0):
                y_true = series[x:x+self.prediction_points][self.columnToPredict].values
            else:
                y_true = np.concatenate((y_true, series[x:x+self.prediction_points][self.columnToPredict].values))

        return forecast, dates, y_true

    def run_fit_predict(self):
        """
            Run, fit and predict function, depending on the parameters inserted
        """
        self.create_trainning_set() 
        
        self.fitEnsemble() 
        
        return self.predict()
    
    def compute_errors(self, forecast, val):
        """
            Compute model errors

            forecast - forecast
            val - validation set
        """

        # Compute Errors (MAE, MSE and MAPE)
        mae_cross_val = mean_absolute_error(val, forecast)
        mse_cross_val = mean_squared_error(val, forecast)
        mape = mean_absolute_percentage_error(val, forecast)

        return mae_cross_val, mse_cross_val, mape
        
    def force_retrain(self): 
        """
            Funtion to force the retrain of the ensemble model
            This function is called when a drift is detected
        """

        if(len(self.forecasts_models_new) <= 7):
            return None
       
        #print("RETRAIN")
        
        self.retrain_periods.append(self.forecasts_models_new_dates[-1])
        
        test_errors_df = self.timeseries_df.copy() #true error
        test_errors_df = test_errors_df[test_errors_df[self.time_column] >= self.forecasts_models_new_dates[0]]
        test_errors_df = test_errors_df[test_errors_df[self.time_column] <= self.forecasts_models_new_dates[-1]]
        test_errors_df = test_errors_df.reset_index(drop=True)
        
        new_extract_features_data_y=[]
        validation_df = pd.DataFrame([])
        for x in range(0, len(test_errors_df), self.prediction_points):
            validation_df = validation_df.append([[test_errors_df[x:x+self.prediction_points][self.columnToPredict].values]])
            new_extract_features_data_y.append(test_errors_df[x:x+self.prediction_points][self.columnToPredict].values.mean())
        
        errors_new_df = pd.DataFrame([], columns=self.forecasts_models_new.columns)      
        count = 0

        for index, row in self.forecasts_models_new.iterrows():
            new_errors = []
            
            val_set = validation_df.iloc[count]
            
            for model_type in self.forecasts_models_new.columns:
                        
                mae_error, mse_error, mape_error = self.compute_errors(row[model_type], val_set[0])
                
                if(self.error_metric == "MAE"):
                    new_errors.append(mae_error)
                elif(self.error_metric == "MSE"):
                    new_errors.append(mse_error)
                else:
                    new_errors.append(mape_error)
        
            errors_new_df = errors_new_df.append(pd.DataFrame([new_errors], columns=self.forecasts_models_new.columns), ignore_index = True)
        
        errors_new_df = self.__calc_best_model(errors_new_df)
        
        self.new_training_set["Best Model"] = errors_new_df["Best Model"].values #self.train_set
        self.forecasts_models_new #self.models_forecast_df
        new_extract_features_data_y #self.extract_features_data_y


        #JOIN SETS
        self.train_set = pd.concat([self.train_set, self.new_training_set], axis=0)
        self.train_set = self.train_set.reset_index(drop=True)
        
        self.models_forecast_df = pd.concat([self.models_forecast_df, self.forecasts_models_new], axis=0)
        self.models_forecast_df = self.models_forecast_df.reset_index(drop=True)

        self.extract_features_data_y = np.concatenate((self.extract_features_data_y, new_extract_features_data_y))

        #TODO: AS
        self.train_set = self.train_set[-700:]
        self.models_forecast_df = self.models_forecast_df[-700:]
        self.extract_features_data_y = self.extract_features_data_y[-700:]



        new_aux = pd.DataFrame(self.train_set["Best Model"].value_counts())
        new_aux = new_aux[new_aux["Best Model"] == 1].index.values
        
        
        models_to_use_aux = self.models_to_use.copy()
        for x in models_to_use_aux:
            if(x not in self.train_set["Best Model"].value_counts().index.values):
                print(f"Removed {x}")
                self.models_to_use.remove(x)   
                self.models_forecast_df.drop([x], axis = 1, inplace=True)

        for x in new_aux:
            print(f"Removed {x}")
            self.models_to_use.remove(x)            
            self.models_forecast_df.drop([x], axis = 1, inplace=True)
            index_ = self.train_set[self.train_set["Best Model"] == x].index[0]
            
            self.train_set=self.train_set.drop(index=(index_))
            self.models_forecast_df=self.models_forecast_df.drop(index=(index_))
            
        self.train_set = self.train_set.reset_index(drop=True)
        self.models_forecast_df = self.models_forecast_df.reset_index(drop=True)
         
        
        self.new_training_set = pd.DataFrame([])
        self.forecasts_models_new = pd.DataFrame([])
        self.forecasts_models_new_dates = []
        
        self.fitEnsemble()

    def print_report(self): 
    
        pd.concat(self.proba).to_csv("proba.csv")
        
        pd.DataFrame(self.features_col).to_csv("features_col.csv")

        pd.DataFrame(self.retrain_periods).to_csv("retrain_periods.csv")

        num = 0
        for x in self.rules_vec:
            x.to_csv(f"rules{num}.csv")
            num = num + 1
            
        print("Num of Retrain Periods: " + str(len(self.retrain_periods)))







