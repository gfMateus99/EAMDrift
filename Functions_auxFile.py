# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 09:47:07 2022

@author: B28069
"""

#%%Imports
import os
import pandas as pd
from datetime import datetime

from datetime import timedelta

import warnings
warnings.filterwarnings("ignore")


#%% Comparing CPU with Tweets

def groupByDay(dataframe_toPlot, splitDates):
    dataframe_toPlot = dataframe_toPlot[dataframe_toPlot['just_date'] >= splitDates["just_date"].min()]
    dataframe_toPlot = dataframe_toPlot[dataframe_toPlot['just_date'] <= splitDates["just_date"].max()]
    dataframe_toPlot['count'] = dataframe_toPlot['just_date'] 
    dataframe_toPlot = dataframe_toPlot.groupby(['just_date'])['count'].count()
    dataframe_toPlot = pd.DataFrame(dataframe_toPlot)
    dataframe_toPlot['count_norm'] = (dataframe_toPlot['count'] - dataframe_toPlot['count'].min()) / (dataframe_toPlot['count'].max() - dataframe_toPlot['count'].min())    
    dataframe_toPlot['count_norm'] = dataframe_toPlot['count_norm']*100
    return dataframe_toPlot

#%%Import funcions
def splitFilesPath(directoryPath, splitWords):
    twitterFilesNames = os.listdir(directoryPath)
    twitterFiles_cleaned_names = []
    twitterFiles_all_names = []
    for file in twitterFilesNames:
        if splitWords in file:
            twitterFiles_cleaned_names.append(directoryPath+file)
        else:
            twitterFiles_all_names.append(directoryPath+file)
    return twitterFiles_cleaned_names, twitterFiles_all_names

def getTwitterData(pathArray):
    data = []
    for path in pathArray:
       custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
       file = pd.read_csv(path, 
                         parse_dates=['created_at'],
                         date_parser=custom_date_parser,
                         low_memory=False)
       file = file.drop(['Unnamed: 0'], axis=1)
       file["just_date"] = file["created_at"].dt.date
       
       data.append(file)
       
    return data

def loadData(layer, timeStep):
    """
        Function that loads data, organize dates and return an dictionary 
        with the requested dataframes
        
        Return dataframe list of servers with the corresponding time step 
        
        layer - layer to analyze
        timeStep - time step
    """
    
    servers_entries = os.listdir('Datacenter_Data/'+layer+'/'+timeStep)
    list_servers = []
    
    for server in servers_entries:
        file = pd.read_csv('Datacenter_Data/'+layer+'/'+timeStep+'/'+server) 
        timeGraph = []        
        for time in file["timestamp"]:
            timeGraph.append(datetime.fromtimestamp(time))            
        file["date"] = timeGraph
        file["hour"] = file["date"].dt.hour
        file["just_date"] = file["date"].dt.date
        file['date'] = correctDates(file)
        if 'Unnamed: 0' in file.columns:   
            file = file.drop(['Unnamed: 0'], axis=1)
        
        list_servers.append(file)

    return list_servers

def load_patterns_data(timeStep):
    """
        Function that loads patterns data, and return an dictionary 
        with the requested dataframes
        
        Return dataframe list of servers with the corresponding time step 
        
        timeStep - time step
    """
    
    servers_entries = os.listdir('Datacenter_Data/Patterns_files/'+timeStep)
    
    list_files = []
    
    for server in servers_entries:
        file = pd.read_csv('Datacenter_Data/Patterns_files/'+timeStep+'/'+server) 
        
        if 'Unnamed: 0' in file.columns:   
            file = file.drop(['Unnamed: 0'], axis=1)
        
        file['date'] = pd.to_datetime(file['date'], format='%Y-%m-%d %H:%M:%S')
        file['date'] = correctDates(file)
        list_files.append(file)

    predictdates = pd.read_csv('Datacenter_Data/Patterns_files/6_hours_toPredictDates.csv')
    predictdates['dates'] = pd.to_datetime(predictdates['dates'], format='%Y-%m-%d %H:%M:%S')
    predictdates['dates2'] = pd.to_datetime(predictdates['dates2'], format='%Y-%m-%d %H:%M:%S')

    return list_files, predictdates

def correctDates(file):
    """
        Function that correct some small errors in dates due due to time zone 
        changes
        
        Return a list with the correct dates
        
        file - file to correct dates
    """
    
    dateSize = len(file)
    firstDate = file.iloc[0]["date"] 
    newDates = []
    newDates.append(firstDate)
    
    for i in range(dateSize-1):
        firstDate = firstDate + timedelta(hours=6)    
        newDates.append(firstDate)    
        
    return newDates
               

