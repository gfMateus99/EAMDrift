# EAMDrift (Interpretable model)

### Preamble
1) Ensemble
2) Based on Mixure of Experts (weights)
3) Interpretable model
4) Real-time re-trainning
5) Allows past co-variates

### Organization of this document

  1) [Repository Structure](#folder_structure)
  2) [EAMDrift model (Documentation)](#EAMDrift_model)
  3) [Example Usage (Tutorial)](#usage_example)
  4) [Run with your own models (Tutorial)](#run_with_your_models)


## <a name="folder_structure"></a> 1. Repository Structure:

<pre>
<b>EAMDrift/</b>  
│  
├─── <b>EAMDrift_model/</b>  
│    ├─── __init__.py
│    ├─── Ensemble_Model_Class.py  
│    ├─── Models.py  
│    ├─── <b>rulefit/</b>  
│         ├─── __init__.py
│         ├─── rulefit.py
│    └─── <b>ModelsDB/</b>  
│         ├─── Models_Interface.py  
│         ├─── ExponentialSmoothingClass.py  
│         ├─── Prophet.py  
│         ├─── StatsForecastAutoARIMA.py  
│         ├─── KalmanForecasterClass.py  
│         └─── Transformer.py  
│  
├─── Run Model.py  
│  
└─── README.md  
</pre>

### Files Descriptions

- **Ensemble_Model_Class.py**

  - This script contains the code to apply anomaly detection methods to data from four sensors (water temperature, specific conductance, pH, dissolved oxygen) at six sites in the Logan River Observatory. 

- **Models.py**

- **Ensemble_Model_Class.py**

- **ModelsDB files**

- **Run Model.py**

## <a name="EAMDrift_model"></a> 2. EAMDrift model (Documentation):

### Prerequisites:

This model depends on the following Python packages:

- darts
- river
- prophet
- numpy
- pandas
- sklearn
- math
- typing
- tqdm
- datetime
- warnings
- logging

### EAMDrift:
# 


<b>class EAMDriftModel(```self,
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
                 n_jobs_: Optional[int] = None```)</b>

<b>Parameters</b>
 - timeseries_df_(object) - Number of time steps to be input to the forecasting module.
 - columnToPredict_(str)
 - time_column_(str)
 - models_to_use_(list)
 - dataTimeStep_(str)
 - trainning_samples_size_(Optional[int])
 - trainning_points_(int)
 - prediction_points_(int)
 - to_extract_features_(bool)

#### Methods
# 

| Method | Description |
| :---:   | :---: |
| [create_trainning_set()](#create_trainning_set) | Split dataframe in subsets to train the ensemble model. |
| [fit()](#fit) | Fit/train the model on one series. | 
| [historical_forecasts()](#historical_forecasts) | Compute the historical forecasts that would have been obtained by this model on the series. |


##### <a name="create_trainning_set"></a> create_trainning_set()
##### <a name="fit"></a> fit()
Train the model with a specific darts.utils.data.TrainingDataset instance. These datasets implement a PyTorch Dataset, and specify how the target and covariates are sliced for training

**Returns:** Fitted model.
**Returns type:** self.

##### <a name="historical_forecasts"></a> historical_forecasts()

## <a name="usage_example"></a> 3. Example Usage (Tutorial):

<pre>
<b>Global folder</b>  
│  
├─── EAMDrift_model  
│  
└─── YOUR_CODE.py
</pre>


**Discover how many jobs you can use to run the model**

Open your shell and run the following command:
```ps
WMIC CPU Get NumberOfCores
```


**1. Imports**
```python
##############################################################################
# Imports
##############################################################################
from EAMDrift_model import EAMDriftModel
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

##############################################################################
# Load files and covariates
##############################################################################
data = pd.read_csv(r'data_hours_organized_reduced.csv')
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S')
covariates_aux = data.copy()
covariates_col = covariates_aux.columns
covariates_aux = covariates_aux.reset_index()
covariates_aux.drop(covariates_col, axis = 1, inplace=True)

```

**2. Run EAMDriftModel**
```python
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
     
```

## <a name="run_with_your_models"></a> 4. Run with your own models (Tutorial):





