# -*- coding: utf-8 -*-
"""

@author: Anonymous

ProphetClass

"""

from EAMDrift_model.ModelsDB.Models_Interface import ModelsInterface

import pandas as pd

import torch.nn as nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel
from darts.utils.timeseries_generation import datetime_attribute_timeseries

import logging
cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.disabled = True

import warnings
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

class LSTMClass(ModelsInterface):
    
    def __init__(self, modelName_: str, train_df_: object, pointsToPredict_: int, columnToPredict_: str, time_column_: str, dataTimeStep_: str, params_: list):
        self.MODEL_NAME = modelName_
        self.points_to_predict = pointsToPredict_
        self.train_df = train_df_.copy()
        self.train_timeseries = TimeSeries.from_dataframe(train_df_, time_column_, columnToPredict_, freq=dataTimeStep_)
        self.dataTimeStep = dataTimeStep_
        self.params = params_ #n_layers, n_dropout, hid_dim, lr, loss   #[1, 0, 150, 0.001, 'MSE'] [2, 0, 150, 0.001, 'l1'] 
        
        self.model = None
        self.transformer = None 
        self.future_covariates = None

    def getModelName(self):
        return self.MODEL_NAME
    
    def gridSearch(self):
        self.run_and_fit_model()
    
    def run_and_fit_model(self):
        
        self.transformer = Scaler()
        trainTransformed = self.transformer.fit_transform(self.train_timeseries)        
        
        self.model = self.__run_lstm_model(self.transformer,
                                        trainTransformed, 
                                        self.train_df,
                                        n_rnn_layers_=self.params[0], 
                                        dropout_=self.params[1], 
                                        hidden_dim_=self.params[2], 
                                        lr=self.params[3],
                                        loss=self.params[4])     
                                                            
    def __run_lstm_model(self, transformer, trainTransformed, train_df, model_="LSTM", hidden_dim_=20, 
                       dropout_=0, batch_size_=16, n_epochs_=300, lr=1e-5, n_rnn_layers_=1, model_name_="Lstm_model", 
                       random_state_=42, training_length_=32, input_chunk_length_=28, loss = "MSE"):
        """
            Return lstm model, mae_cross_val, mse_cross_val with given definitions
            
            transformer - trasnformer created to scale the data
            trainTransformed - data to be trained
            valTransformed - data to be validates
            series - series of all data 
            date_split - date to split train and validation sets
        """
            
        # 1: Create month and year covariate series
        year_series = datetime_attribute_timeseries(
            pd.date_range(start=trainTransformed.start_time(), freq=trainTransformed.freq_str, periods=1600),
            attribute="year",
            one_hot=False,
        )
        
        year_series = Scaler().fit_transform(year_series)
        month_series = datetime_attribute_timeseries(
            year_series, attribute="month", one_hot=True
        )
        
        covariates_trans = year_series.stack(month_series)
        #cov_train, cov_val = covariates.split_before(trainTransformed.end_time())
        
        self.future_covariates = covariates_trans
        
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
        model = RNNModel(
            model=model_,
            hidden_dim=hidden_dim_,
            dropout=dropout_,
            batch_size=batch_size_,
            n_epochs=n_epochs_,
            optimizer_kwargs={"lr": lr},
            model_name=model_name_,
            loss_fn= ll,
            log_tensorboard=False,
            random_state=random_state_,
            training_length=training_length_,
            input_chunk_length=input_chunk_length_,
            n_rnn_layers=n_rnn_layers_,
            force_reset=True,
            save_checkpoints=False,
            pl_trainer_kwargs={"callbacks": [my_stopper]}
        )
        
        # 3: Fit the model
        model.fit(
            trainTransformed,
            future_covariates=covariates_trans,
            verbose=False
        )
        
        return model
       
    def predict(self):
        y_pred = self.model.predict(self.points_to_predict, future_covariates=self.future_covariates)
        return self.transformer.inverse_transform(y_pred)._xa.values.flatten()








