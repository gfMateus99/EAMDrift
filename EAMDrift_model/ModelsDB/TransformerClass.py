# -*- coding: utf-8 -*-
"""
ProphetClass
"""

from EAMDrift_model.ModelsDB.Models_Interface import ModelsInterface

import pandas as pd

import torch.nn as nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries

from darts.models.forecasting.transformer_model import TransformerModel


import logging
cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.disabled = True

import warnings
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

class TransformerClass(ModelsInterface):
    
    def __init__(self, modelName_: str, train_df_: object, pointsToPredict_: int, columnToPredict_: str, time_column_: str, dataTimeStep_: str, params_: list):
        self.MODEL_NAME = modelName_
        self.points_to_predict = pointsToPredict_
        self.train_df = train_df_.copy()
        self.train_timeseries = TimeSeries.from_dataframe(train_df_, time_column_, columnToPredict_, freq=dataTimeStep_)
        self.dataTimeStep = dataTimeStep_
        self.params = params_ # 0.15, 0.0001, 'MSE'
        
        self.model = None
        self.transformer = None 
        self.past_covariates = None

    def getModelName(self):
        return self.MODEL_NAME
    
    def gridSearch(self):
        self.run_and_fit_model()
    
    def run_and_fit_model(self):
        
        self.transformer = Scaler()
        trainTransformed = self.transformer.fit_transform(self.train_timeseries)        
        
        self.model = self.__run_transformer_model(self.transformer, 
                                                  trainTransformed, 
                                                  dropout_=float(self.params[0]), 
                                                  lr=float(self.params[1]),
                                                  loss = self.params[2],
                                                  activation_='relu',
                                                  random_state_=42,
                                                  train = self.train_df)                   
                                                            


    def __run_transformer_model(self, transformer, trainTransformed, 
                       model_="TransformerModel", dropout_=0, batch_size_=16, n_epochs_=300, 
                       lr=1e-5, model_name_="Lstm_model", random_state_=42, 
                       input_chunk_length_=28, loss = "MSE", output_chunk_length_=4, 
                       activation_='relu',train=None, val=None, covariates_array=[]):
        """
            Return TransformerModel model, mae_cross_val, mse_cross_val with given definitions
            
            transformer - trasnformer created to scale the data
            trainTransformed - data to be trained
            valTransformed - data to be validates
            series - series of all data 
            date_split - date to split train and validation sets
        """
            
        # 1: Create month and year covariate series
        year_series = datetime_attribute_timeseries(
            pd.date_range(start=trainTransformed.start_time(), freq=trainTransformed.freq_str, periods=len(train)+(240)),
            attribute="year",
            one_hot=False,
        )
        year_series = Scaler().fit_transform(year_series)
        month_series = datetime_attribute_timeseries(
            year_series, attribute="month", one_hot=True
        )
        covariates = year_series.stack(month_series)
        
        if (len(covariates_array)!=0):
            ram_series = pd.concat([train, val], ignore_index=True)
            ram_series = TimeSeries.from_dataframe(ram_series, 'date', covariates_array, freq='6H')  
            ram_series = Scaler().fit_transform(ram_series)
            covariates = covariates.stack(ram_series)
    
        self.past_covariates = covariates

        ll = nn.MSELoss()
        if loss == "l1": 
            ll = nn.L1Loss()
            
        # stop training when validation loss does not decrease more than 0.05 (`min_delta`) over
        # a period of 5 epochs (`patience`)
        my_stopper = EarlyStopping(
            monitor="train_loss",
            patience=5,
            min_delta=0.05,
            mode='min',
        )
        
        # 2: Define the model   
        model = TransformerModel(
            input_chunk_length=input_chunk_length_, 
            output_chunk_length=output_chunk_length_, 
            d_model=64, 
            nhead=4, 
            optimizer_kwargs={"lr": lr},
            num_encoder_layers=3, 
            num_decoder_layers=3, 
            dim_feedforward=512, 
            dropout=dropout_, 
            activation=activation_,
            loss_fn= ll,
            batch_size=batch_size_,
            n_epochs=n_epochs_,
            model_name=model_name_,
            random_state=random_state_,
            log_tensorboard=False,
            force_reset=True,
            save_checkpoints=False,
            pl_trainer_kwargs={"callbacks": [my_stopper]}
        )
        
        # 3: Fit the model
        model.fit(
            trainTransformed,
            past_covariates=covariates,
            verbose=False
        )
        
        return model
            

    def predict(self):
        y_pred = self.model.predict(self.points_to_predict, past_covariates=self.past_covariates)
        return self.transformer.inverse_transform(y_pred)._xa.values.flatten()








