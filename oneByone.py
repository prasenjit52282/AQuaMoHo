import argparse
import numpy as np
import pandas as pd
from library.constants import *
from library.helper import scale
from library.models.rnn import RNN
from library.models.rf import RandomForest
from library.utils import read_dataset_from_files,read_csv

parser = argparse.ArgumentParser(description='Training for different experiments with rf and rnn')
parser.add_argument('--restore', help='Required decision to restore weights',action='store_true')
args = parser.parse_args()
restore=args.restore

#-------------------------------------------------------------------------------------------------
#Durgapur
train_div=\
['./Data/Dgp/Device_1_merged.csv',
 './Data/Dgp/Device_2_merged.csv',
 './Data/Dgp/Device_4_merged.csv'
]

test_div='./Data/Dgp/Device_3_merged.csv'

#Random Forest
dataset=read_dataset_from_files(train_div,features,label,window=None)
test_dataset=read_csv(test_div,features,label,window=None)

groups=pd.DataFrame(pd.Series(dataset["ts"]).apply(lambda e:e[:7]),columns=["ts"]).groupby('ts')
months=list(groups.groups.keys())

l=[]
for i,month in enumerate(months):
    if i==0:
        #initial
        X_train=dataset["X"][np.array(groups.groups[month])]
        y_train=dataset["y"][np.array(groups.groups[month])]
        X_test=test_dataset["X"]
        y_test=test_dataset["y"]
    else:
        #add month data
        X_train=np.concatenate([X_train,dataset["X"][np.array(groups.groups[month])]],axis=0)
        y_train=np.concatenate([y_train,dataset["y"][np.array(groups.groups[month])]],axis=0)
        
    X_train_scaled,X_test_scaled,_=scale(X_train,X_test)
    model=RandomForest()
    model.train(X_train_scaled,X_test_scaled,y_train,y_test,epochs=None,batch_size=None)
    l.append(model.metrics)   
    
df=pd.DataFrame(l)
df.to_csv("./logs/exp/dgp_rf_month_1by1.csv",index=False)


#RNN
dataset=read_dataset_from_files(train_div,features,label,window=window)
test_dataset=read_csv(test_div,features,label,window=window)

groups=pd.DataFrame(pd.Series(dataset["ts"]).apply(lambda e:e[:7]),columns=["ts"]).groupby('ts')
months=list(groups.groups.keys())

l=[]
for i,month in enumerate(months):
    if i==0:
        #initial
        X_train=dataset["X"][np.array(groups.groups[month])]
        y_train=dataset["y"][np.array(groups.groups[month])]
        X_test=test_dataset["X"]
        y_test=test_dataset["y"]
    else:
        #add month data
        X_train=np.concatenate([X_train,dataset["X"][np.array(groups.groups[month])]],axis=0)
        y_train=np.concatenate([y_train,dataset["y"][np.array(groups.groups[month])]],axis=0)
        
    X_train_scaled,X_test_scaled,_=scale(X_train,X_test)
    
    X_train_scaled=X_train_scaled.reshape(-1,window,len(features))
    X_test_scaled=X_test_scaled.reshape(-1,window,len(features))
    
    model=RNN(checkpoint_filepath=f'./logs/model/dgp_rnn_1by1_{i}',seed=seed,restore=restore)
    model.train(X_train_scaled,X_test_scaled,y_train,y_test,epochs=epochs,batch_size=batch_size)
    l.append(model.metrics)   
    
df=pd.DataFrame(l)
df.to_csv("./logs/exp/dgp_rnn_month_1by1.csv",index=False)

#-------------------------------------------------------------------------------------------------
#Delhi
train_div=\
['./Data/Delhi/Device_18_merged.csv',
 './Data/Delhi/Device_20_merged.csv',
 './Data/Delhi/Device_22_merged.csv',
 './Data/Delhi/Device_23_merged.csv',
 './Data/Delhi/Device_25_merged.csv',
 './Data/Delhi/Device_28_merged.csv',
 './Data/Delhi/Device_30_merged.csv',
 './Data/Delhi/Device_33_merged.csv',
 './Data/Delhi/Device_34_merged.csv',
 './Data/Delhi/Device_35_merged.csv',
 './Data/Delhi/Device_37_merged.csv'
]

test_div='./Data/Delhi/Device_24_merged.csv'

#Random Forest
dataset=read_dataset_from_files(train_div,features,label,window=None)
test_dataset=read_csv(test_div,features,label,window=None)

groups=pd.DataFrame(pd.Series(dataset["ts"]).apply(lambda e:e[:7]),columns=["ts"]).groupby('ts')
months=list(groups.groups.keys())

l=[]
for i,month in enumerate(months):
    if i==0:
        #initial
        X_train=dataset["X"][np.array(groups.groups[month])]
        y_train=dataset["y"][np.array(groups.groups[month])]
        X_test=test_dataset["X"]
        y_test=test_dataset["y"]
    else:
        #add month data
        X_train=np.concatenate([X_train,dataset["X"][np.array(groups.groups[month])]],axis=0)
        y_train=np.concatenate([y_train,dataset["y"][np.array(groups.groups[month])]],axis=0)
        
    X_train_scaled,X_test_scaled,_=scale(X_train,X_test)
    model=RandomForest()
    model.train(X_train_scaled,X_test_scaled,y_train,y_test,epochs=None,batch_size=None)
    l.append(model.metrics)   
    
df=pd.DataFrame(l)
df.to_csv("./logs/exp/delhi_rf_month_1by1.csv",index=False)


#RNN
dataset=read_dataset_from_files(train_div,features,label,window=window)
test_dataset=read_csv(test_div,features,label,window=window)

groups=pd.DataFrame(pd.Series(dataset["ts"]).apply(lambda e:e[:7]),columns=["ts"]).groupby('ts')
months=list(groups.groups.keys())

l=[]
for i,month in enumerate(months):
    if i==0:
        #initial
        X_train=dataset["X"][np.array(groups.groups[month])]
        y_train=dataset["y"][np.array(groups.groups[month])]
        X_test=test_dataset["X"]
        y_test=test_dataset["y"]
    else:
        #add month data
        X_train=np.concatenate([X_train,dataset["X"][np.array(groups.groups[month])]],axis=0)
        y_train=np.concatenate([y_train,dataset["y"][np.array(groups.groups[month])]],axis=0)
        
    X_train_scaled,X_test_scaled,_=scale(X_train,X_test)
    
    X_train_scaled=X_train_scaled.reshape(-1,window,len(features))
    X_test_scaled=X_test_scaled.reshape(-1,window,len(features))
    
    model=RNN(checkpoint_filepath=f'./logs/model/delhi_rnn_1by1_{i}',seed=seed,restore=restore)
    model.train(X_train_scaled,X_test_scaled,y_train,y_test,epochs=epochs,batch_size=batch_size)
    l.append(model.metrics)   
    
df=pd.DataFrame(l)
df.to_csv("./logs/exp/delhi_rnn_month_1by1.csv",index=False)