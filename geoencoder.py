import urllib2
import json
from IPython import embed
import pandas as pd
import numpy as np
import threading
from multiprocessing import Pool
import time

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

rawdata=pd.read_csv('taxigo.txt',header=None,names=['id','time','lon','lat'],skip_blank_lines=True)
    #if(folder==2):break
rawdata=rawdata.dropna()
rawdata=rawdata.drop('time',axis=1)
rawdata = rawdata[ (rawdata.lon>lon_min) & (rawdata.lon<lon_max) & (rawdata.lat>lat_min) & (rawdata.lat<lat_max)]

hist = list(pd.read_csv('geo.csv', header=None, index_col=0).index)

idxs = rawdata.index.copy()
i = list(idxs)

for h in hist:
    i.remove(h)


def work(sel):
    xy = rawdata.ix[sel]
    try:
    	req = "http://api.map.baidu.com/geocoder/v2/?callback=renderReverse&location=%.6f,%.6f&output=json&pois=0&coordtype=wgs84ll&ak=NY7GUfWOHM3GRX2m75M1SXDAEYHG0Qmt" % (xy['lat'],xy['lon'])
    	raw = urllib2.urlopen(req).read()
    	d = json.loads(raw[29:-1])
    	street = d['result']['addressComponent']['street']
        newdata = '%d,%s,%f,%f' % (sel, street, xy['lat'], xy['lon'])
        return newdata
    except:
        return None

p = 300

f = open('geo.csv','a') 
for _ in range(len(i)):
    #threads=[]
    t=time.time()
    pool = Pool(p)
    sel = np.random.choice(i, p, replace=False)
    for s in sel:
        i.remove(s)
    buf = pool.map(work,sel)
    pool.close()
    pool.join()
    #t = threading.Thread(target=work, args=(sel,))
    #threads.append(t)
    #t.start()

    print "joining"
    #main_thread = threading.currentThread()
    #for t in threading.enumerate():
    #    if t is not main_thread:
    #        t.join()

    try:
	f = open('geo.csv','a') 
    	for b in buf:
            print b
            if b is not None: f.write((b+'\n').encode('utf-8'))
	f.close()
    except:
        print "can't write"
        exit(0)
    print p, 'takes', time.time() - t
