
# coding: utf-8

# In[9]:

from __future__ import division
import os
from collections import Counter
import gc
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np


# In[10]:

ProcNum = 20
duration = 24*3600
square_len = 50

camera_NUM = 1000

meter_per_lat = 111300
meter_per_log = 85300
#北京位于东经115.7°—117.4°，北纬39.4°—41.6°
lon_min=115
lon_max=118
lat_min=39
lat_max=42


# In[ ]:

rawdata=pd.DataFrame([])
mypath=os.getcwd()+'\\taxigo.txt'
#files=[join(mypath,f) for f in listdir(mypath) if isfile(join(mypath,f))]
#for i in files:
rawdata=pd.read_csv(mypath,header=False,names=['id','time','lon','lat'],skip_blank_lines=True)
    #if(folder==2):break
rawdata=rawdata.dropna()
rawdata=rawdata.drop('time',axis=1)
rawdata = rawdata[ (rawdata.lon>lon_min) & (rawdata.lon<lon_max) & (rawdata.lat>lat_min) & (rawdata.lat<lat_max)]


# In[30]:

def readData():
    return np.array([rawdata.id,square_decode(rawdata.lon,rawdata.lat)]).T


# In[32]:

def dataPreprocess(data_list):
    data_dict = {}
    for item in data_list:
        data_dict[str(item[1])]={}

    for item in data_list:
        data_dict[str(item[1])][item[0]]=0

    for item in data_list:
        data_dict[str(item[1])][item[0]] += 1
    return data_dict


# In[33]:

def fun(x):
    return duration/(x+1)
  
def reward(taxi_count, taxi_freq):
    output = 0
    for item in taxi_freq.values():
        output += fun(item)
    output += (taxi_count-len(taxi_freq))*duration
    return output/taxi_count

def square_decode(lon, lat):
    max_col =  int(((lat_max-lat_min)*meter_per_lat + square_len - 1)/square_len)
    row = (((lon_max-lon)*meter_per_log + square_len - 1)/square_len).astype(int)
    col = (((lat-lat_min)*meter_per_lat + square_len - 1)/square_len).astype(int)

    result =  (row-1)*max_col + col
    return result

def square_encode(sID):
    sID = int(sID)
    max_col =  int(((lat_max-lat_min)*meter_per_lat + square_len - 1)/square_len)
    row = int((sID + max_col  - 1)/max_col)
    col = sID%max_col
    return ((col*square_len)/meter_per_lat+lat_min, lon_max-(row*square_len)/meter_per_log)


# In[ ]:

ct = 0

f = open(os.getcwd()+'\\result.txt','w')
raw_data = readData()
print 'data loading finished'
taxi_count = len({x[0] for x in raw_data})
#print taxi_count
data_dict = dataPreprocess(raw_data)

print 'data preprocessing finished'
del raw_data
gc.collect()

data_dict_keys = data_dict.keys()
data_dict_values = data_dict.values()
min_step = ('', {})

for jjj in xrange(camera_NUM):
    reward_values = []

    for i in xrange(len(data_dict_keys)):
        reward_values.append(reward(taxi_count,dict(Counter(data_dict_values[i])+Counter(min_step[1]))))
    #print reward_values
    min_square = reward_values.index(min(reward_values))

    min_step = (data_dict_keys[min_square], dict(Counter(data_dict_values[min_square])+Counter(min_step[1])))
    min_reward = min(reward_values)
    ct += 1
    print ct,square_encode(data_dict_keys[min_square])[0], square_encode(data_dict_keys[min_square])[1], min_reward/60, len(min_step[1].keys()),taxi_count
    f.writelines(','.join([str(ct),
        str(square_encode(data_dict_keys[min_square])[1]),str(square_encode(data_dict_keys[min_square])[0]),
        str(min_reward/60), str(len(min_step[1].keys())), str(taxi_count)])+'\n')

    data_dict_keys.pop(min_square)
    data_dict_values.pop(min_square)
f.close()



# In[8]:

import heatmap

hm = heatmap.Heatmap()

pts = []
f = open("result.txt", 'r')

for line in f:
    cor=line[:-1].split(',')
    pts.append((float(cor[1]), float(cor[0])))
pts = pts[:10]
f.close()
hm.heatmap(pts, scheme='classic',dotsize=30,opacity=1000, size=(1024,1024), area=((lon_min,lat_min),
    (lon_max,lat_max)))

hm.saveKML("camer.kml")

