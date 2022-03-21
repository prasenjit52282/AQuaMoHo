import glob
import numpy as np
import pandas as pd
from datetime import datetime

def timecluster(x):
    if x >=0 and x <= 6:
        return 0
    elif x >= 7 and x <= 9:
        return 1
    elif x>=10 and x<=16:
        return 2
    elif x >= 17 and x <= 21:
        return 3
    elif x >= 22:
        return 0

def season(x):
    if x >=1 and x <= 2:
        return 0
    elif x >= 3 and x <= 7:
        return 1
    elif x>=8 and x<=9:
        return 2
    elif x >= 10:
        return 0

def read(f):
    df=pd.read_csv(f)
    df['AQI']=df.AQI.apply(lambda e:5 if e==6 else e)-1
    df['hour'] = df.timestamp.apply(lambda e: int(e.split(" ")[1]))
    df['timecluster'] = df['hour'].apply(lambda e:timecluster(e))
    df['month'] = df.timestamp.apply(lambda e: int(e.split("-")[1]))
    df['season'] = df['month'].apply(lambda e:season(e))
    df['dayofweek'] = df.timestamp.apply((lambda e: datetime.strptime(e.split(" ")[0],'%Y-%m-%d').date().weekday()))
    return df

def get_features_labels(df,features,label):
    feat=df[features].copy()
    lbl=df[label].copy()
    return feat,lbl

def do_windowing(feat,lbl,window_size):
    X=[];y=[];i=0
    total_size=feat.shape[0]
    while i+window_size<=total_size:
        ser=feat.iloc[i:i+window_size,:].values
        aqi=lbl.iloc[i:i+window_size].values[-1]
        if not (np.isnan(ser).mean()>0):
            X.append(ser)
            y.append(aqi)
        i+=1
    y_arr=np.array(y)
    X_arr=np.array(X)
    return X_arr,y_arr

def read_csv_raw(f,features,label):
    df=read(f)
    X,y=get_features_labels(df,features,label)
    return {'X':X,'y':y}

def read_csv_filtered(f,features,label):
    df=read(f).dropna()
    X,y=get_features_labels(df,features,label)
    return {'X':X,'y':y}

def read_csv_windowing(f,features,label,window):
    data=read_csv_raw(f,features,label)
    X,y=do_windowing(data['X'],data['y'],window)
    feat_size=X.shape[-1]
    return {'X':X.reshape(-1,window*feat_size),'y':y,
            'feat_size':feat_size,"window_size":window}

def read_csv(f,features,label,window=None):
    if window is None:
        return read_csv_filtered(f,features,label)
    else:
        return read_csv_windowing(f,features,label,window)
    
def read_dataset(loc,features,label,window=None):
    files=glob.glob(loc)
    X=[];y=[]
    for f in files:
        data=read_csv(f,features,label,window)
        X.append(data['X'])
        y.append(data['y'])
    if window is None:return {'X':np.concatenate(X,axis=0),'y':np.concatenate(y,axis=0)}
    else:return {'X':np.concatenate(X,axis=0),'y':np.concatenate(y,axis=0),
                 'feat_size':data['feat_size'],'window_size':data['window_size']}
    
def read_dataset_from_files(files,features,label,window=None):
    X=[];y=[]
    for f in files:
        data=read_csv(f,features,label,window)
        X.append(data['X'])
        y.append(data['y'])
    if window is None:return {'X':np.concatenate(X,axis=0),'y':np.concatenate(y,axis=0)}
    else:return {'X':np.concatenate(X,axis=0),'y':np.concatenate(y,axis=0),
                 'feat_size':data['feat_size'],'window_size':data['window_size']}