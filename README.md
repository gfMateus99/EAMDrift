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
  4) [Output example](#output)


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
├─── Data  
│    └─── DATA_FILE.csv
│  
└─── README.md  
</pre>

## <a name="EAMDrift_model"></a> 2. EAMDrift model (Documentation):

### Prerequisites:

This model was tested in Anaconda with Python (version 3.9.12) and depends on the following packages:

- darts (version 0.20.0)
- river (version 0.9.0)
- prophet (version 1.1.1)
- numpy (version 1.21.5)
- pandas (version 1.4.2)
- scikit-learn (version 1.0.2)
- tqdm (version 4.64.0)
- tsfresh (version 0.19.0)
- joblib (version 1.1.0)


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
 - **timeseries_df_(object) -** Dataframe with timeseries data.
 - **columnToPredict_(str) -** Column in timeseries_df_ to predict.
 - **time_column_(str) -** Column in timeseries_df_ with date time
 - **models_to_use_(list) -** List of models to use (at least two models are necessary)
 - **dataTimeStep_(str) -** Time step of time_column_ (i.e. "6H", "1H", ...).
 - **covariates_: Optional[object] -** Dataframe with covariates data.
 - **categorical_covariates_ : Optional[list] -** Categorical covariates in covariates_ dataframe.
 - **covariates_ma_: Optional[int] -** Observation window for covariates at each prediction (an observation of 7 corresponds to using the last seven covariates lines).
 - **error_metric_: Optional[str] -** Metric to select the best models. The default is "MAPE."
 - **trainning_samples_size_: Optional[int] -** Number of samples in trainningset. The model will try to make the maximum number of possible subsets if no value is given.
 - **trainning_points_: int -** Observation window for target variable at each prediction (an observation of 7 corresponds to using the last seven training set lines). The default value is 150.           
 - **fit_points_size_: int -** Number of points the last models will use to predict. The default value is 150.
 - **prediction_points_: int -** Number of points to predict. The default value is 4.
 - **to_extract_features_: bool -** Extract features from dataset. In case of False, the model will use covariates to predict. In case of no existence of covariates, an error will occur. The default value is True. 
 - **use_dates_: bool -** To use dates covariates to predict. The default value is True.    
 - **selectBest_: int -** Number of models to use to predict points. If None, all models of models_to_use_ will be used. The default value is None.
 - **to_retrain_: Optional[bool] -** To retrain automatically. The default value is True.
 - **n_jobs_: Optional[int] -** Number of jobs to use. The default value is 4.            

#### Methods
# 

| Method | Description |
| :---:   | :---: |
| **[create_trainning_set()](#create_trainning_set)** | Split dataframe in subsets to train the ensemble model. |
| **[fitEnsemble()](#fit)** | Fit/train the model on target serie. | 
| **[historical_forecasts()](#historical_forecasts)** | Compute the historical forecasts that this model in a real-time situation would have obtained. |
| **[predict()](#predict)** | Predict the next x points. |
| **[change_timeseries()](#change_timeseries)** | Change the timeseries that is beeing analyzed. The trainning set will remain the same until a retrain is needed. |
| **[add_and_predict()](#add_and_predict)** | Add more values to timeseries to predict the next ones. |
| **[run_fit_predict()](#run_fit_predict)** | Automatically runs all functions to predict (create_trainning_set() and predict()). |
| **[compute_errors()](#compute_errors)** | Compute the errors given a true and forecast array. |
| **[force_retrain()](#force_retrain)** | Force the retrain of the model. |
| **[print_report()](#print_report)** | Save files with info about model run (probabilities, rules, features selected). |


## <a name="usage_example"></a> 3. Example Usage (Tutorial):

The dataset used in this test is the Electric Power Consumption (EPC) **[1]**. It measures the electric power usage in different houses in the zone of Paris, France, and for this test, just one house was chosen. The data has a 1-minute step for nearly four years but was aggregated in hours, giving us 35063 entries (to be faster to compute, in this test, we use the last 2500 entries of the total 35063). As electric consumption can be related to weather, we used data from AWOS sensors available in **[2]** to be used as covariates.

**[1]** Hebrail, Georges, Berard, Alice. (2012). Individual household electric power consumption. UCI Machine Learning Repository. Accessed: 10/02/2023 [Online]. Available: doi.org/10.24432/C58K54

**[2]** Iowa State University. AWOS Global METAR. Accessed: 05/03/2023 [Online]. Available: mesonet.agron.iastate.edu/request/download.phtml


**Create a file in the same folder of EAMDrift_model:**
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

# Load data
data = pd.read_csv(r'Data/LD2011_2014(MT_330).csv')
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S')
data = data.drop(["Unnamed: 0"], axis=1)

# Prepare covariates 
covariates_aux = data.copy()
covariates_col = covariates_aux.columns
covariates_aux.drop(["date", "MT_330"], axis = 1, inplace=True)
```

**2. Run EAMDriftModel**
```python
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
                                   #selectBest_=1, #None selects all models
                                   to_retrain_=True,
                                   n_jobs_=6
                                   )
    
    # Make Trainning set
    trainning_set_init, self_ = ensemble_model.create_trainning_set()     
    
    # Train ensemble method
    rules = ensemble_model.fitEnsemble() 
    
    # Predict - predict just the next points
    #prediction = ensemble_model.predict() # Predict the next points_to_predict points
    
    # Predict with historical forecasts
    forecast, dates, y_true = ensemble_model.historical_forecasts(dataframe[index:], 
                                                                  forecast_horizon = int((len(dataframe)-index)/points_to_predict), 
                                                                  covariates = covariates_aux[index:]) 
    # Compute errors
    print(ensemble_model.compute_errors(forecast[:-points_to_predict]*mean_y+std_y, y_true*mean_y+std_y))
     
```

## <a name="output"></a> 4. Output Example:

**1. Console output**

The output of the console explains what the model is doing. It can be split into 3 phases that will be described next.

Training phase - In this phase, the model creates a new training set and will the ensemble with that data. In this case, the model removed the ExponentialSmoothing as it was not chosen as the best at least two times.
```python
Splitting dataframe: 100%|██████████| 112/112 [00:00<00:00, 527.54it/s]
Feature Extraction: 100%|██████████| 14/14 [00:13<00:00,  1.00it/s]
Note: 246 features selected from a total of 789
Inserting accuracys (6 models): 100%|██████████| 6/6 [48:35<00:00, 486.00s/it]
Removed ExponentialSmoothing
Ensemble Model accuracy: 0.8571428571428571
```

Predicting phase - Here, the model will compute the predictions over time. Each new line corresponds to a new training zone with the respective new accuracy of the ensemble model.
```python
 12%|█▏        | 10/83 [12:21<1:30:07, 74.07s/it]
 
 Ensemble Model accuracy: 0.7235772357723578
 24%|██▍       | 20/83 [23:49<1:14:43, 71.17s/it]

 Ensemble Model accuracy: 0.7819548872180451
 66%|██████▋   | 55/83 [1:04:14<31:08, 66.73s/it]
 
 Ensemble Model accuracy: 0.7321428571428571
 84%|████████▍ | 70/83 [1:21:57<14:16, 65.87s/it]
 
 Ensemble Model accuracy: 0.7377049180327869
100%|██████████| 83/83 [1:37:17<00:00, 70.33s/it]
```

Compute the errors - Finally, the model will call the compute_errors() function to compute the prediction errors. The user will receive the MSE, MAE, and MAPE error.
```python
Mape: 0.17892600316561688
```

**2. Files output**

Finally if the user evoke the print_report() function, the model will output the number of retrains until the moment that was asked and a set of files will be save in the root folder.

**Output:**
```python
Num of Retrain Periods: 4
```

**Saved files:**

**`proba.csv`** - Contribution of each model to each prediction.

**`features_col.csv`** - Statistics extracted.

**`retrain_periods.csv`** - Date of each retrain period.

**`rules{num}.csv`** - Set of files, each corresponding to each set of rules generated in each retrain period. In these case the model would ooutput 5 files, starting with **`num=0`**.




