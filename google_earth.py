# encoding:utf-8
#北京位于东经115.7°—117.4°，北纬39.4°—41.6°
lon_min=115.7
lon_max=117.4
lat_min=39.4
lat_max=41.6

import heatmap

hm = heatmap.Heatmap()

pts = []
f = open("result.txt", 'r')

for line in f:
    cor=line[:-1].split(' ')
    pts.append((float(cor[2]), float(cor[1])))
pts = pts[:10]
f.close()

hm.heatmap(pts, scheme='classic',dotsize=30,opacity=1000, size=(1024,1024), area=((lon_min,lat_min),
    (lon_max,lat_max)))

hm.saveKML("camer.kml")

