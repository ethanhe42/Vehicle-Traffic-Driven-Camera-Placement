import urllib2
import json
from IPython import embed
raw = urllib2.urlopen("http://api.map.baidu.com/geocoder/v2/?callback=renderReverse&location=39.983424,116.322987&output=json&pois=1&ak=NY7GUfWOHM3GRX2m75M1SXDAEYHG0Qmt").read()
d = json.loads(raw[29:-1])
street = d['result']['addressComponent']['street']
