
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
from datetime import datetime
from datetime import timedelta
from dateutil.parser import parse


# In[2]:

def timeTrans(time):
    return datetime.strptime(time,'%Y-%m-%d %H:%M:%S')


# #北京位于东经115.7°—117.4°，北纬39.4°—41.6°
# lon_min=115
# lon_max=118
# lat_min=39
# lat_max=42

# In[3]:

#北京位于东经115.7°—117.4°，北纬39.4°—41.6°
lon_min=115.7
lon_max=117.4
lat_min=39.4
lat_max=41.6


# In[4]:

ProcNum = 20
duration = 24*3600
square_len = 50

camera_NUM = 10000

meter_per_lat = 111300
meter_per_log = 85300
log = 'logs/'
names = ['MUV','MVT','OITR','meet','AUI']
verbosenames=['S1 MUV','S2 MVT','S3 OITR','S4 meet','S5 AUI']


# In[5]:

nlat=int(((lat_max-lat_min)*meter_per_lat + square_len - 1)/square_len)
nlon=int(((lon_max-lon_min)*meter_per_log + square_len - 1)/square_len)
print "latitude",int(((lat_max-lat_min)*meter_per_lat + square_len - 1)/square_len)
print "longitude",int(((lon_max-lon_min)*meter_per_log + square_len - 1)/square_len)


# In[6]:

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
def overlap(a,b):
    return set(a).intersection(set(b))


# In[7]:

rawdata=pd.DataFrame([])
mypath='taxigo.txt'
#files=[join(mypath,f) for f in listdir(mypath) if isfile(join(mypath,f))]
#for i in files:
rawdata=pd.read_csv(mypath,header=None,names=['id','time','lon','lat'],skip_blank_lines=True)
    #if(folder==2):break
rawdata=rawdata.dropna()
rawdata = rawdata[ (rawdata.lon>lon_min) & (rawdata.lon<lon_max) & (rawdata.lat>lat_min) & (rawdata.lat<lat_max)]
rawdata.time=rawdata.time.map(timeTrans)


##### time delta

# In[8]:

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


# In[9]:

rawdata['date'] = rawdata.time
rawdata.time=timespan.time
rawdata.time[rawdata.time>3600]=14

del shift
del shifted
del timespan

rawdata.id=rawdata.id.astype(int)

def similar():
    import urllib2
    import json
    codes = ['NY7GUfWOHM3GRX2m75M1SXDAEYHG0Qmt', '8BPhSwUz2BBrcw8LqwQXXqkTVZG2zDx5'][::-1]
    codeidx = 0
    def work(start, end):
        req='http://api.map.baidu.com/direction/v1?mode=driving&origin'+    '=%.6f,%.6f&destination=%.6f,%.6f&origin_region=北京&destination_region=北京&output=json&coordtype=wgs84ll&ak=%s' % (start[0][0],start[1][0],end[0][0],end[1][0],codes[codeidx])
        raw = urllib2.urlopen(req).read()
        ans = json.loads(raw)
        lonset=[]
        bset=[]
        dist=0
        for i in ans['result']['routes'][0]['steps']:
            lat = i['stepDestinationLocation']['lat']
            lon = i['stepDestinationLocation']['lng']
            dist += i['distance']
            lonset.append(lon)
            bset.append(lat)
        return np.array(bset),np.array(lonset),dist


    first=10
    plt.scatter(np.array(rawdata[rawdata.id==9754].lon)[:first], np.array(rawdata[rawdata.id==9754].lat)[:first])
    plt.show()
    dfs = []
    names = ['MUV','MVT','OITR','meet','AUI']
    for i in names:
        dfs.append(pd.read_csv(i+'_final.csv'))
    def compute_ratio(car, strategy=-1):
        blockseq = square_decode(rawdata[rawdata.id==car].lon, rawdata[rawdata.id==car].lat)
        hits = overlap(blockseq, dfs[strategy].Region)
        previous=None
        blocks=[]
        alldist=1
        ratio=0
        for a in blockseq:
            blocks.append(a)
            if a in hits:
                if previous == None:
                    previous = a
                else:
                    dest=square_encode(np.array([a]))
                    start=square_encode(np.array([previous]))
                    latset,lonset,dist=work(start,dest)
                    seq=square_decode(lonset,latset)
                    alldist+=dist
                    if len(seq) <= 2:
                        r = 1
                    else:
                        blocks=set(blocks)
                        cnt=0
                        for i in seq:
                            if i in blocks:
                                cnt+=1
                        r = cnt/len(seq)
                    ratio+=dist*r
                    previous = a
                blocks=[]
        ratio/=alldist
        print(car, ratio)
    for i in range(10000):
        try:
            compute_ratio(i,-5)
        except:
            continue

def setoverlaps():
    dfs = []
    names = ['MUV','MVT','OITR','meet','AUI']
    for i in names:
        dfs.append(pd.read_csv(i+'.csv').Region)
    pidx=0
    for i in range(len(dfs)):
        for j in range(i+1, len(dfs)):
            y=[]
            x = range(0, 10000,100)
            for k in x:
                 y.append(len(set(dfs[i][:k]).intersection(set(dfs[j][:k]))))
            pidx+=1
            ax = plt.subplot(2,5, pidx)
            ax.set_title(names[i]+' '+names[j])
            ax.plot(x,y)
    t=' '.join(verbosenames)
    plt.title=t
    plt.show()

def day2daystats():
    df=[]
    for n in names:
        dfs = [pd.read_csv(log+n+'.csv.'+str(i)).Region for i in range(2, 9)]
        lap=[]
        for i,j in zip(dfs, dfs[1:]):
            a=set(i)
            b=set(j)
            a.update(b)
            lap.append(len(overlap(i,j))/len(a))
        df.append(lap)
    pd.DataFrame(df, index=verbosenames).T.plot()
    plt.savefig('day2day.eps')


def speed_hist():
    dist_shift = rawdata.drop(['time'], axis=1)
    dist_shifted = dist_shift.shift().fillna(0)
    dist_span = dist_shift - dist_shifted
    dist_span['lat'][dist_span.id!=0]=0
    dist_span['lon'][dist_span.id!=0]=0
    dist_delta = ((dist_span['lat']*meter_per_lat)**2 + (dist_span['lon']*meter_per_log)**2)**.5
    speed_delta = (dist_delta / rawdata.time).fillna(0)
    speed_delta[speed_delta > 100] = 100
    speed_delta[rawdata.time == 14] = 0
    speed_delta[rawdata.time == 0] = 0
    speed_delta.hist(color='grey', bins=100)
    plt.show()
    len(speed_delta[speed_delta == 100])
    #rawdata[rawdata.id == speed_delta[speed_delta == 100].index[0]].plot(kind='scatter',x='lon',y='lat')
    rawdata[rawdata.id == speed_delta[speed_delta == 100].index[0]][:1000].plot(x='lon',y='lat')
    plt.show()

    print 'valid squares',len(data_traffic.region.unique())/(nlon*nlat)


##### drop time

# In[11]:

#rawdata=rawdata.drop('time',axis=1)


# In[12]:

# def readData():
#     return np.array([rawdata.id,square_decode(rawdata.lon,rawdata.lat)]).T

# def dataPreprocess(data_list):
#     data_dict = {}
#     for item in data_list:
#         data_dict[str(item[1])]={}

#     for item in data_list:
#         data_dict[str(item[1])][item[0]]=0

#     for item in data_list:
#         data_dict[str(item[1])][item[0]] += 1
#     return data_dict


## maximize traffic (MVT

# In[10]:

def getMVT(data, suffix=''):
    data_traffic=data.join(pd.Series(square_decode(data.lon,data.lat),name='region'))
    traffic=data_traffic.groupby('region').time.sum()
    traffic.sort(ascending=False)
    #traffic.hist(bins=1000)
    #plt.show()
    id_cnts=data_traffic.drop('time',axis=1).groupby(by=['region','id']).count()
    id_cnts.lat=1
    id_cnts.columns=['traffic','init']
    id_cnts=id_cnts.drop('traffic',axis=1)
    IDs=set(data_traffic.id.astype(int))
    IDs=pd.Series(0,index=IDs)
    AUI=[]
    VCR=0
    cnt=0
    for i in traffic.index:
        minRegion=int(i)

        IDs[np.array(id_cnts.init[minRegion].index,dtype=int)]=1
        # region , dVCR, VCR , UCR
        VCR+=traffic[minRegion]
        AUI.append([minRegion,traffic[minRegion],VCR,IDs.sum()])

        cnt+=1
        if cnt==camera_NUM:
            break
    AUI=pd.DataFrame(AUI,columns=['Region','dVCR','VCR','UCR'])
    AUI.VCR/=traffic.sum()
    AUI.UCR=AUI.UCR.astype(float)/len(IDs)
    AUI['loc']=AUI.Region.apply(square_encode)
    AUI.to_csv(log+'MVT.csv'+suffix)


## Maximum Unique Vehicles (MUV

# In[11]:

def getMUV(data, suffix=''):
    data_traffic=data.join(pd.Series(square_decode(data.lon,data.lat),name='region'))
    traffic=data_traffic.groupby('region').time.sum()
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

        minRegion=int(id_cnts.sum(level=0).argmax())


        IDs[np.array(id_cnts[minRegion].index,dtype=int)]=0
        #update
        id_cnts=            pd.Series(np.array(IDs[id_cnts.index.get_level_values('id')]),index=id_cnts.index)


        # region , dVCR, VCR , UCR
        VCR+=traffic[minRegion]
        AUI.append([minRegion,traffic[minRegion],VCR,len(IDs)-IDs.sum()])
        #print(minRegion)

    AUI=pd.DataFrame(AUI,columns=['Region','dVCR','VCR','UCR'])
    AUI.VCR/=traffic.sum()
    AUI.UCR=AUI.UCR.astype(float)/len(IDs)
    AUI['loc']=AUI.Region.apply(square_encode)
    AUI.to_csv(log+'MUV.csv'+suffix)


## Mixed Maximum Strategy (MMS

# In[12]:

def getMMS(data, suffix=''):
    data_traffic=data.join(pd.Series(square_decode(data.lon,data.lat),name='region'))
    traffic=data_traffic.groupby('region').time.sum()

    for alpha in range(60*1600,60*1601,1200):


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
        AUI.to_csv(log+'MMS'+str(int(alpha/60))+'.csv'+suffix)


## Minimum Mean AUIs (Average Unique-Camera-Hit Intervals)

# In[13]:

def getAUI(data, suffix=''):
    data_traffic=data.join(pd.Series(square_decode(data.lon,data.lat),name='region'))
    traffic=data_traffic.groupby('region').time.sum()
    id_cnts=data_traffic.drop('time',axis=1).groupby(by=['region','id']).count()
    id_cnts.lat=1
    id_cnts.columns=['traffic','init']
    IDs=set(data_traffic.id.astype(int))
    AUI=[]
    IDs=pd.Series(0,index=IDs)
    inits=pd.Series(1,IDs.index)

    # only difference between OICR
    id_cnts.traffic=1
    VCR=0
    for i in range(camera_NUM):
        minRegion=int(((1/(id_cnts.traffic+id_cnts.init))-1/id_cnts.init).sum(level=0).argmin())

        #update inversers

        inits[id_cnts.init[minRegion].index]+=id_cnts.traffic[minRegion]
        id_cnts.init=pd.Series(np.array(inits[id_cnts.index.get_level_values('id')]),id_cnts.index)


        #set id to cover
        dUCR=len(np.array(id_cnts.init[minRegion].index,dtype=int))
        IDs[np.array(id_cnts.init[minRegion].index,dtype=int)]=1

        #cal traffic
        dVCR=traffic[minRegion]
        VCR+=dVCR

        AUI.append([minRegion,dVCR,VCR,dUCR,IDs.sum()])
        #print minRegion,dVCR,VCR,dUCR,IDs.sum()
        id_cnts.traffic[int(minRegion)]=0
    AUI=pd.DataFrame(AUI,columns=['Region','dVCR','VCR','dUCR','UCR'])
    AUI.VCR/=traffic.sum()
    AUI.UCR=(AUI.UCR).astype(float)/len(IDs)
    AUI['loc']=AUI.Region.apply(square_encode)
    AUI.to_csv(log+'AUI.csv'+suffix)


## Out-Camera to In-Camera Time Ratio (OITR

# In[17]:

def getOITR(data, suffix=''):
    data_traffic=data.join(pd.Series(square_decode(data.lon,data.lat),name='region'))
    traffic=data_traffic.groupby('region').time.sum()
    id_cnts=data_traffic.groupby(by=['region','id']).sum()
    id_cnts=id_cnts.drop('lon',axis=1)
    id_cnts.lat=1
    id_cnts.columns=['traffic','init']
    IDs=set(data_traffic.id.astype(int))
    AUI=[]
    IDs=pd.Series(0,index=IDs)
    inits=pd.Series(1,IDs.index)
    idx=[8180782,6795037]
    for i in idx:
        pass
        #print ((1/(id_cnts.traffic+id_cnts.init))-1/id_cnts.init).sum(level=0)[i]
    VCR=0
    for i in range(camera_NUM):

        minRegion=int(((1/(id_cnts.traffic+id_cnts.init))-1/id_cnts.init).sum(level=0).argmin())

        #update inversers
        idx=np.array(id_cnts.init[minRegion].index)
        inits[idx]+=id_cnts.traffic[minRegion]
        id_cnts.init=pd.Series(np.array(inits[id_cnts.index.get_level_values('id')]),id_cnts.index)


        #set id to cover
        dUCR=len(np.array(id_cnts.init[minRegion].index,dtype=int))
        IDs[np.array(id_cnts.init[minRegion].index,dtype=int)]=1

        #cal traffic
        dVCR=traffic[minRegion]
        VCR+=dVCR

        AUI.append([minRegion,dVCR,VCR,dUCR,IDs.sum()])
        #print minRegion,dVCR,VCR,dUCR,IDs.sum()
        id_cnts.traffic[minRegion]=0

    #     if minRegion==9077097:
    #         break
    AUI=pd.DataFrame(AUI,columns=['Region','dVCR','VCR','dUCR','UCR'])
    AUI.VCR/=traffic.sum()
    AUI.UCR=AUI.UCR.astype(float)/len(IDs)
    AUI['loc']=AUI.Region.apply(square_encode)
    AUI.to_csv(log+'OITR.csv'+suffix)


## S6

# In[19]:

def getmeet(data, suffix=''):
    data_traffic=data.join(pd.Series(square_decode(data.lon,data.lat),name='region'))
    traffic=data_traffic.groupby('region').time.sum()
    id_cnts=data_traffic.groupby(by=['region','id']).count()
    id_cnts=id_cnts.drop('lon',axis=1)
    id_cnts.lat=1
    id_cnts.columns=['traffic','init']
    IDs=set(data_traffic.id.astype(int))
    AUI=[]
    IDs=pd.Series(0,index=IDs)
    inits=pd.Series(1,IDs.index)
    traffic=data_traffic.groupby('region').time.sum()
    traffic.sort(ascending=False)
    #traffic[:10000].sum()/traffic.sum()
    VCR=0
    for i in range(camera_NUM):

        minRegion=int(((1/(id_cnts.traffic+id_cnts.init))-1/id_cnts.init).sum(level=0).argmin())

        #update inversers
        idx=np.array(id_cnts.init[minRegion].index)
        inits[idx]+=id_cnts.traffic[minRegion]
        id_cnts.init=pd.Series(np.array(inits[id_cnts.index.get_level_values('id')]),id_cnts.index)


        #set id to cover
        dUCR=len(np.array(id_cnts.init[minRegion].index,dtype=int))
        IDs[np.array(id_cnts.init[minRegion].index,dtype=int)]=1

        #cal traffic
        dVCR=traffic[minRegion]
        VCR+=dVCR

        AUI.append([minRegion,dVCR,VCR,dUCR,IDs.sum()])
        #print minRegion,dVCR,VCR,dUCR,IDs.sum()
        id_cnts.traffic[minRegion]=0

    AUI=pd.DataFrame(AUI,columns=['Region','dVCR','VCR','dUCR','UCR'])
    AUI.VCR/=traffic.sum()
    AUI.UCR=AUI.UCR.astype(float)/len(IDs)
    AUI['loc']=AUI.Region.apply(square_encode)
    AUI.to_csv(log+'meet.csv'+suffix)


# # day 2 day


funcs = [getMUV, getMVT, getOITR, getmeet, getAUI]


def day2day_algo():
    def whichday(n):
        return lambda x: x.day == n
    for d in range(2, 9):
        data = rawdata[rawdata.date.map(whichday(d))].drop('date',1)
        for f in funcs:
            f(data, '.'+str(d))


def withinday():
    cnt=9
    late=4
    orgdate=datetime(2008,2,2)
    while True:
        t=datetime(2008,2,2+cnt//24,cnt%24)
        tp=datetime(2008,2,2+(cnt+late)//24,(cnt+late)%24)
        if t == datetime(2008,2,9):
            break
        else:
            cnt+=1
        suffix=int((t-orgdate).total_seconds()//3600)
        suffix='.h'+str(suffix)
        dateidx=(rawdata.date>=t) & (rawdata.date<=tp)
        if sum(dateidx)==0:
            print suffix,"continue"
        else:
            print suffix
            data=rawdata[dateidx].drop('date',1)
            for f in funcs:
                f(data, suffix)

withinday()
## add VUH & VIT

#data_traffic=rawdata.join(pd.Series(square_decode(rawdata.lon,rawdata.lat),name='region'))
#gb=data_traffic.groupby('region').id.unique()
#gb=gb.apply(len)
#gb=pd.DataFrame(gb)
#gb.columns=['VUH']
#meet=data_traffic.groupby('region').id.count()
#meet=pd.DataFrame(meet)
#meet.columns=['meet']
#Nvehicles=len(data_traffic.id.unique())
#
#
#
#import matplotlib.pyplot as plt
#import numpy as np
#import matplotlib as mpl
#filenames=['MUV','MVT','OITR','meet','AUI']
##,'MMS1600'
#strategy=['S1','S2','S3','S4','S5','S6']
#metrics=['VCR','UCR','meanVIT','meet','meanVUH']
#metricsName=['VCR','UCR','mean VIT (min)','mean VCH','mean VUH']
#position=['upper left','lower right','upper left','upper left','upper left']
#
#markers = [
#
#'v', # point
#'o', # circle
#'^', # triangle up
#'s',
#'d', # thin_diamond
#'>',
#]
#color=[
#    'b' ,        #blue
#    'g' ,       # green
#    'r'  ,      # red
#    'c'  ,     #  cyan
#    'm'  ,      # magenta
#    'y'  ,     #  yellow
#    'k'  ,    #   black
#    'w'  ,     #  white
#]
#figFormat=['eps','jpg']
#
#
## In[55]:
#
#font = {'family' : 'normal',
#        'weight' : 'normal',
#        'size'   : 16}
#
#mpl.rc('font', **font)
#
#
## In[56]:
#
#def AddMean(name):
#    addmean=pd.read_csv(name+'.csv',index_col=0)
#    addmean['meanVIT']=addmean.dVCR.cumsum()/Nvehicles/60.0 #/np.array(range(1,1001))
#    addmean=pd.merge(addmean,gb,how='left',left_on='Region',right_index=True)
#    addmean=pd.merge(addmean,meet,how='left',left_on='Region',right_index=True)
#    addmean.meet=addmean.meet.cumsum()/Nvehicles
#    addmean['meanVUH']=addmean.VUH.cumsum()/Nvehicles#/np.array(range(1,1001))
#    addmean.to_csv(name+'_final.csv')
#
#
## In[57]:
#
#for i in filenames:
#    AddMean(i)
#
#
### plot
#
## In[550]:
#
#csvs=[]
#for i in filenames:
#    csvs.append(pd.read_csv(i+'_final.csv',index_col=0))
#
#
##### fig2 special
#
## In[552]:
#
#metric=1
#mark=0
#if metric==1:
#    markersize=[14,9,9,9,9,9]
#else:
#    markersize=[9,9,9,9,9,9]
#for i in csvs:
#      #  print i
#
#    i[metrics[1]].plot(c=color[mark],
#                            marker=markers[mark],
#                            markevery=int(camera_NUM/5),
#                            markersize=markersize[mark],
#                            markeredgecolor=color[mark],
#                            linewidth=2.5)
#    mark+=1
#
#plt.legend(strategy,loc=position[metric])
##    plt.title(metrics[metric])
#plt.xlabel('N')
#plt.ylabel(metricsName[metric])
#plt.xticks(range(0,camera_NUM+1,int(camera_NUM/5)))
##     plt.ylim([0,0.5])
##     plt.xlim([0,20])
#for f in figFormat:
#    plt.savefig(metrics[metric]+'.'+f,format=f)
#
## this is another inset axes over the main axes
#a = plt.axes([0.25, 0.15, .4, .6])
#mark=0
#for i in csvs:
#    plt.plot(i[metrics[1]][:1000],
#            marker=markers[mark],
#            markevery=int(camera_NUM/10/5),
#            markeredgecolor=color[mark])
#    mark+=1
#
##plt.plot(t[:len(r)], r)
##plt.title('Impulse response')
#plt.ylim(0.9, 1)
##plt.
##plt.xticks([])
#plt.yticks(np.arange(0.9,1.01,0.02))
#for f in figFormat:
#    plt.savefig(metrics[metric]+'.'+f,format=f)
#
#plt.show()
#
#
## In[579]:
#
#csvs[1]['dVCR'].plot()
#plt.show()
#
#
#### plot metrics
#
## In[551]:
#
#for metric in range(len(metrics)):
#    mark=0
#    if metric==1:
#        markersize=[14,9,9,9,9,9]
#    else:
#        markersize=[9,9,9,9,9,9]
#    for i in csvs:
#          #  print i
#        i[metrics[metric]].plot(c=color[mark],
#                                marker=markers[mark],
#                                markevery=int(camera_NUM/5),
#                                markersize=markersize[mark],
#                                markeredgecolor=color[mark],
#                                linewidth=2.5)
#        mark+=1
#    plt.legend(strategy,loc=position[metric])
##    plt.title(metrics[metric])
#    plt.xlabel('N')
#    plt.ylabel(metricsName[metric])
#    plt.xticks(range(0,camera_NUM+1,int(camera_NUM/5)))
##     plt.ylim([0,0.5])
##     plt.xlim([0,20])
#    for f in figFormat:
#        plt.savefig(metrics[metric]+'.'+f,format=f)
#    plt.show()
#
#
###### box plot
#
## In[390]:
#
#data_traffic=rawdata.join(pd.Series(square_decode(rawdata.lon,rawdata.lat),name='region'))
#
#
## In[581]:
#
#VIT=[]
#VUH=[]
#VCH=[]
#def inbox(i):
#    return i in set(box.region)
#
#def boxer(name):
#    box=pd.read_csv(name+'_final.csv',index_col=0)
#    #print box
#
#    #print boxer
#    box=pd.DataFrame(pd.Series(box.Region,name='region'))
#    dropdata=data_traffic.copy()
#    merged=pd.merge(dropdata,box)
#
#    aVIT=merged.groupby(['id']).time.sum()/60.0
#    VIT.append(aVIT)
#    print name,'VIT','STD=',aVIT.std(),'median=',aVIT.median()
#    aVUH=merged.groupby(['id']).region.unique().apply(len)
#    VUH.append(aVUH)
#    print name,'VUH','STD=',aVUH.std(),'median=',aVUH.median()
#    aVCH=merged.groupby(['id']).time.count()
#    VCH.append(aVCH)
#    print name,'VCH','STD=',aVCH.std(),'median=',aVCH.median()
#
#
## In[592]:
#
#VIT=[]
#VUH=[]
#VCH=[]
#for i in filenames:
#    boxer(i)
#
#
## In[593]:
#
#all_data=[]
#all_data.append(VIT)
#all_data.append(VUH)
#all_data.append(VCH)
#
#
## In[394]:
#
#boxData=['VIT (min)','VUH','VCH']
#
#
## In[618]:
#
#for i in range(len(boxData)):
#
#    bplot = plt.hist(all_data[i],
#                    bins=n_bins,
#                   normed=1,
#                   histtype='step',
#                   cumulative=True
#                       )   # vertical box aligmnent
#    #plt.xticks([y+1 for y in range(len(all_data[i]))], strategy)
#
#    #plt.ylabel(boxData[i])
#    plt.xlabel(boxData[i])
#    #plt.xlim([0,8000])
#
#    #t = plt.title(boxData[i])
#    for f in figFormat:
#        plt.savefig('CDF_'+boxData[i]+'.'+f,format=f)
#    plt.show()
#
#
## In[590]:
#
#for i in range(len(boxData)):
#
#    bplot = plt.boxplot(all_data[i],
#            notch=False, # box instead of notch shape
#            sym='rs',    # red squares for outliers
#            vert=True,
#            showfliers=False
#
#                       )   # vertical box aligmnent
#    plt.xticks([y+1 for y in range(len(all_data[i]))], strategy)
#
#    plt.ylabel(boxData[i])
#    plt.xlabel('strategy ID')
#
#    for components in bplot.keys():
#        for line in bplot[components]:
#            line.set_color('black')     # black lines
#
#    #t = plt.title(boxData[i])
#    for f in figFormat:
#        plt.savefig('box_'+boxData[i]+'.'+f,format=f)
#    plt.show()
#
#
### heatmap
#
## In[530]:
#
#for i in csvs[0][:563]['loc']:
#    print i.strip('()').split(', ')
#    break
#
#
## In[611]:
#
##北京位于东经115.7°—117.4°，北纬39.4°—41.6°
#lon_min=115.7
#lon_max=117.4
#lat_min=39.4
#lat_max=41.6
#
#import heatmap
#
#hm = heatmap.Heatmap()
#
#pts = []
## f = open("result.txt", 'r')
#
#for i in csvs[1][:563]['loc']:
#    cor=i.strip('()').split(', ')
##     print float(cor[2])
##     break
#    pts.append((float(cor[1]), float(cor[0])))
##pts = pts[:10]
## f.close()
#
#
## In[542]:
#
#pts
#
#
## In[612]:
#
#hm.heatmap(pts, scheme='classic',dotsize=30,opacity=1000, size=(1024,1024), area=((lon_min,lat_min),
#    (lon_max,lat_max)))
#
#hm.saveKML("camer.kml")

