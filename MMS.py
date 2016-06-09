
# coding: utf-8



# In[1]:

from __future__ import division
import os
from collections import Counter
import gc
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:

from datetime import datetime
from datetime import timedelta
from dateutil.parser import parse


# In[3]:
# alpha configuration seconds
start=60*1600
end=60*1601
step=1200
def timeTrans(time):
    return datetime.strptime(time,'%Y-%m-%d %H:%M:%S')


#北京位于东经115.7°—117.4°，北纬39.4°—41.6°
lon_min=115.7
lon_max=117.4
lat_min=39.4
lat_max=41.6


# In[6]:

ProcNum = 20
duration = 24*3600
square_len = 50

camera_NUM = 10000

meter_per_lat = 111300
meter_per_log = 85300


# In[7]:

nlat=int(((lat_max-lat_min)*meter_per_lat + square_len - 1)/square_len)
nlon=int(((lon_max-lon_min)*meter_per_log + square_len - 1)/square_len)
print "latitude",int(((lat_max-lat_min)*meter_per_lat + square_len - 1)/square_len)
print "longitude",int(((lon_max-lon_min)*meter_per_log + square_len - 1)/square_len)


# In[8]:

rawdata=pd.DataFrame([])
mypath=os.getcwd()+'\\taxigo.txt'
#files=[join(mypath,f) for f in listdir(mypath) if isfile(join(mypath,f))]
#for i in files:
rawdata=pd.read_csv(mypath,header=False,names=['id','time','lon','lat'],skip_blank_lines=True)
    #if(folder==2):break
rawdata=rawdata.dropna()
rawdata = rawdata[ (rawdata.lon>lon_min) & (rawdata.lon<lon_max) & (rawdata.lat>lat_min) & (rawdata.lat<lat_max)]
rawdata.time=rawdata.time.map(timeTrans)


# #### time delta

# In[9]:

def time2int(t):
    return t.item().total_seconds()

#get time delta
shift=rawdata.sort_index(by=['id','time'])
shift=shift.drop(['lon','lat'],axis=1)
shift.id=shift.id.astype(int)
shifted=shift.shift().fillna(0)
timespan=(shift-shifted)
# delta time 2 int (seconds)
timespan.time=timespan.time/ np.timedelta64(1, 's')
timespan.time[timespan.id!=0]=14


# In[10]:

rawdata.time=timespan.time
rawdata.time[rawdata.time>3600]=14

del shift
del shifted
del timespan


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
    sID = sID
    max_col =  int(((lat_max-lat_min)*meter_per_lat + square_len - 1)/square_len)
    row = ((sID + max_col  - 1)/max_col).astype(int)
    col = np.mod(sID,max_col)
    
    return ((col*square_len)/meter_per_lat+lat_min, lon_max-(row*square_len)/meter_per_log)



# #Mixed Maximum Strategy (MMS

# In[152]:



data_traffic=rawdata.join(pd.Series(square_decode(rawdata.lon,rawdata.lat),name='region'))
traffic=data_traffic.groupby('region').time.sum()
for alpha in range(start,end,step):

    
    id_cnts=data_traffic.drop('time',axis=1).groupby(by=['region','id']).count()
    id_cnts.lat=0
    id_cnts.columns=['traffic','init']
    IDs=set(data_traffic.id.astype(int))
    AUI=[]
    IDs=pd.Series(1,index=IDs)


    id_cnts.traffic=1
    id_cnts=id_cnts.drop('init',axis=1)
    #uv.sort(ascending=False)
    id_cnts=id_cnts.traffic
    AUI=[]
    VCR=0
    for i in range(camera_NUM):

        minRegion=int((traffic+alpha*id_cnts.sum(level=0)).argmax())


        IDs[np.array(id_cnts[minRegion].index,dtype=int)]=0
        #update
        id_cnts=pd.Series(np.array(IDs[id_cnts.index.get_level_values('id')]),index=id_cnts.index)
        

        # region , dVCR, VCR , UCR
        VCR+=traffic[minRegion]
        AUI.append([minRegion,traffic[minRegion],VCR,len(IDs)-IDs.sum()])
        traffic[minRegion]=0

    AUI=pd.DataFrame(AUI,columns=['Region','dVCR','VCR','UCR'])
    traffic=data_traffic.groupby('region').time.sum()
    
    AUI.VCR/=traffic.sum()
    AUI.UCR=AUI.UCR.astype(float)/len(IDs)
    AUI['loc']=AUI.Region.apply(square_encode)
    AUI.to_csv('MMS'+str(int(alpha/60))+'.csv')
    print 'finish',alpha

