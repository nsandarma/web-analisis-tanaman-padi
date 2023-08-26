import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


dataset = pd.read_csv('data_lengkap.csv')

def getXandY(provinsi):
    data = dataset.loc[dataset['Provinsi'] == provinsi]
    X = data[['Luas Panen',"Curah hujan","Kelembapan","Suhu rata-rata"]]
    y = data[['Produksi']]
    return X,y
    
def getScaler(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler
    
    