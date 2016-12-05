import urllib2
import json
from IPython import embed
import pandas as pd

rawdata=pd.read_csv(mypath,header=False,names=['id','time','lon','lat'],skip_blank_lines=True)
    #if(folder==2):break
rawdata=rawdata.dropna()
rawdata=rawdata.drop('time',axis=1)
rawdata = rawdata[ (rawdata.lon>lon_min) & (rawdata.lon<lon_max) & (rawdata.lat>lat_min) & (rawdata.lat<lat_max)]
embed()
raw = urllib2.urlopen("http://api.map.baidu.com/geocoder/v2/?callback=renderReverse&location=39.983424,116.322987&output=json&pois=1&ak=NY7GUfWOHM3GRX2m75M1SXDAEYHG0Qmt").read()
d = json.loads(raw[29:-1])
street = d['result']['addressComponent']['street']

