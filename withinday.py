import pandas as pd
import matplotlib.pyplot as plt
from IPython import embed
log = 'logs/'
names = ['MUV','MVT','OITR','meet','AUI']
verbosenames=['S1 MUV','S2 MVT','S3 OITR','S4 meet','S5 AUI']
def overlap(a,b):
    return set(a).intersection(set(b))
df=[]
for n in names:
    dfs = [pd.read_csv(log+n+'.csv.h'+str(i)).Region for i in range(10, 103)]
    lap=[]
    for i,j in zip(dfs, dfs[1:]):
        a=set(i)
        b=set(j)
        a.update(b)
        lap.append(float(len(overlap(i,j)))/len(a))
    df.append(lap)
pd.DataFrame(df, index=verbosenames).T.plot()
plt.savefig('withinday.eps')