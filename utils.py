
meter_per_lat = 111300
meter_per_log = 85300
lon_min=115
lon_max=118
lat_min=39
lat_max=42

def square_decode(lon, lat,square_len = 50):
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
