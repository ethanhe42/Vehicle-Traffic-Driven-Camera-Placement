import os
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
block = square_decode(rawdata['lon'], rawdata['lat'],square_len=square_len )
embed()




block
pd.unique?
rawdata.join?
rawdata.join(block)
block.name='block'
newdf = rawdata.join(block)
newdf
newdf.groupby(block)
newdf.groupby(block).unique()
newdf['id'].groupby('block')
newdf['id'].groupby(block)
newdf['road'].groupby(block)
roadgbblock = newdf['road'].groupby(block)
roadgbblock.nunique?
roadgbblock.nunique??
roadgbblock.unique?
roadgbblock.nunique()
pd.unique?
pd.unique([1,1,2])
pd.nunique([1,1,2])
a=pd.Series([1,1,2])
a.nunique()
newdf['block']==32502656
newdf[newdf['block']==32502656]
newdf = newdf.dropna()
roadgbblock = newdf['road'].groupby(newdf['block'])
roadgbblock.nunique()
roadgbblock.nunique().hist()
plt.savefig('50.png')
import matplotlib.pyplot as plt
plt.savefig('50.png')
%save -f tmp.py
%history -f tmp.py
