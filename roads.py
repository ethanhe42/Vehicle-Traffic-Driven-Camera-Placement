import os
import matplotlib.pyplot as plt
from utils import *
from IPython import embed
from collections import Counter
import gc
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np

ProcNum = 20
duration = 24*3600
square_len = 50

camera_NUM = 1000

meter_per_lat = 111300
meter_per_log = 85300
lon_min=115
lon_max=118
lat_min=39
lat_max=42
mypath = 'geo.csv'
rawdata=pd.read_csv(mypath,header=None,names=['id','road','lat','lon'])#,skip_blank_lines=True)
outs=None
risc=None
allcsvpath = 'allgeo.csv'
allcsv=[]

for square_len in [1000,500,100,80,50,40,30,20,10][::-1]:
    block = square_decode(rawdata['lon'], rawdata['lat'],square_len=square_len )
    block.name=str(square_len)
    newdf = rawdata.join(block)
    newdf = newdf.dropna()
    roadgbblock = newdf['road'].groupby(newdf[block.name])
    tmpr = roadgbblock.nunique()
    tmp = tmpr.value_counts()
    if 1:
        allrest =  square_decode(np.array([115]),np.array([42]), square_len=square_len)[0]
        rest = allrest - tmp.sum()
        allcsv.append([square_len, rest,allrest, float(rest)/allrest])
        
    tmp.name=block.name
    tmpr.name=block.name
    if outs is None:
        outs = pd.DataFrame(tmp)
        risc = pd.DataFrame(tmpr)
    else:
        outs = outs.join(tmp, how='outer')
        risc = risc.join(tmpr, how='outer')

pd.DataFrame(allcsv, columns=['square', 'empty','all','percent']).to_csv(allcsvpath)

risc.boxplot(whis=100000)
plt.savefig('50box.eps')
#embed()

#outs.to_csv()
cu = 5
newouts = outs[outs.index<cu]
newouts.loc[cu]=outs[outs.index>=cu].sum()
newouts.plot.bar()
jlt.savefig('50.eps')
