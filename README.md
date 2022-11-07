# EAMDrift (Interpretable model)

### Preamble
1) Ensemble
2) Based on Mixure of Experts (weights)
3) Interpretable model
4) Real-time re-trainning
5) Allows past, future, and static co-variates

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
│    ├─── Ensemble_Model_Class.py  
│    ├─── Models.py  
│    └─── <b>ModelsDB/</b>  
│         ├─── Models_Interface.py  
│         ├─── ExponentialSmoothingClass.py  
│         ├─── Prophet.py  
│         ├─── StatsForecastAutoARIMA.py  
│         ├─── KalmanForecasterClass.py  
│         └─── Other_Model.py  
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


<b>class EAMDriftModel(```timeseries_df_, columnToPredict_, time_column_, models_to_use_, dataTimeStep_,
                       trainning_samples_size_ = None,
                       trainning_points_ = 150,
                       prediction_points_ = 4,
                       to_extract_features_ = True```)</b>

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

**1. Imports**
```python
#Import Folder
from EAMDrift_model.Ensemble_Model_Class import EAMDriftModel

#Import dataset

```

**2. Run EAMDriftModel**
```python
if __name__ == '__main__':

    dataframe = pl_layer_6_hours[0].copy(
    )[["date", "CPU utilization (average)"]]
    
    models = ["KalmanForecaster", "ExponentialSmoothing", "Prophet"]  # models to use

    ensemble_model = EAMDriftModel(timeseries_df_=dataframe,
                                   columnToPredict_="CPU utilization (average)",
                                   time_column_="date",
                                   models_to_use_=models,
                                   dataTimeStep_="6H",
                                   error_metric_="MAPE",
                                   #trainning_samples_size_ = 100,
                                   trainning_points_=150,
                                   prediction_points_=4,
                                   to_extract_features_=True,
                                   n_jobs_=6)
                                   
    # Create trainning set
    trainning_dataframe_index, trainning_dataframe, errors = ensemble_model.create_trainning_set()
    
    #Fit and predict
     
```

## <a name="run_with_your_models"></a> 4. Run with your own models (Tutorial):





