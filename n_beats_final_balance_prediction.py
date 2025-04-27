#imports
import os
import pandas as pd
import numpy as np

#colab installations
!pip install pytorch_forecasting pytorch_lighning torch
!pip install NeuralForecast

from neuralforecast.core import NeuralForecast
from neuralforecast.models import NBEATS
from sklearn.preprocessing import MinMaxScaler

from google.colab import drive
drive.mount('/content/drive')
sample_data_path = '/content/drive/MyDrive/Colab Notebooks'
files=os.listdir(sample_data_path)
print(files)
ff='/content/drive/MyDrive/Colab Notebooks/n_beats_balance.xlsx'
df=pd.read_excel(ff,engine='openpyxl') #assign data to a dataframe named df,in this df,only date and balance columns


#data visualization
print(df.head()) #print first 5 rows of the dataframe
print(df.info()) #print info about the dataframe
print(df.dtypes)

df["Date"] = pd.to_datetime(df["Date"]) #convert date column to datetime format

#sorting the dataframe by date
df = df.sort_values(by="Date") #sort the dataframe by date column
df = df.reset_index(drop=True) #reset the index of the dataframe
print(df.head()) #print first 5 rows of the dataframe

#Rename colums and create dataframe df_nf

df_nf=df.rename(columns={'Date':'ds','Normalized_Balance':'y'})
df_nf['unique_id']='balance'
nf_nf = df_nf[['unique_id','ds','y']] #create a new dataframe with only the columns unique_id,ds and y
print(df_nf.head()) #print first 5 rows of the dataframe



#define model with horizon
horizon =30 #for 7 days ahead prediction

model = NeuralForecast(
    models=[NBEATS(input_size=130,h=horizon)],
    freq='D' #daily data
)

#Fit the model
model.fit(df_nf)

future = model.predict()

print( future.head()) #print first 5 rows of the predictions dataframe
print( future.info()) #print info about the predictions dataframe

original_min=960.0
original_max=75138.99

future["Predicted_Balance"] = future["NBEATS"] * (original_max - original_min) + original_min

print( future.head()) #print first 5 rows of the predictions dataframe



print(future)