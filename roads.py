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
for square_len in [1000,500,100,80,50,40,30,20,10]:
    block = square_decode(rawdata['lon'], rawdata['lat'],square_len=square_len )
    block.name=str(square_len)
    newdf = rawdata.join(block)
    newdf = newdf.dropna()
    roadgbblock = newdf['road'].groupby(newdf[block.name])
    tmp = roadgbblock.nunique().value_counts()
    tmp.name=block.name
    if outs is None:
        outs = pd.DataFrame(tmp)
    else:
        outs = outs.join(tmp)
outs.to_csv()
cu = 5
newouts = outs[outs.index<cu]
newouts.loc[cu]=outs[outs.index>=cu].sum()
embed()
newouts.plot.bar()
plt.savefig('50.png')
