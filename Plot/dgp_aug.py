import glob
import sys
import argparse
sys.path.append("../")
import numpy as np
import pandas as pd
from library.constants import *
from library.models.rf import RandomForest
from library.models.rnn import RNN
from library.helper import scale
from library.utils import read_dataset_from_files,read_csv

parser = argparse.ArgumentParser(description='Training for different experiments with rf and rnn')
parser.add_argument('--restore', help='Required decision to restore weights',action='store_true')
args = parser.parse_args()
restore=args.restore

Div_files=glob.glob("../Data/Dgp/*")
np.random.seed(seed)
test_set=np.random.choice(Div_files,size=3,replace=False)[[2,1,0]]
train_set=np.array(list(set(Div_files)-set(test_set)))


#Random forest
dataset=read_dataset_from_files(train_set,features,label,window=None)
X_train=dataset["X"]
y_train=dataset["y"]

l=[]
for i,f in enumerate(test_set):
    print(i,f)
    test_data=read_csv(f,features,label,window=None)
    X_test=test_data["X"].values
    y_test=test_data["y"].values
    X_train_scaled,X_test_scaled,_=scale(X_train,X_test)
    model=RandomForest()
    model.train(X_train_scaled,X_test_scaled,y_train,y_test,epochs=None,batch_size=None)
    l.append(model.metrics)
    X_train=np.concatenate([X_train,X_test],axis=0)
    annot=model.pred_fn(X_test_scaled)
    y_train=np.concatenate([y_train,annot],axis=0)

df=pd.DataFrame(l)
df.to_csv("./logs/exp/dgp_rf_aug.csv",index=False)


#RNN
dataset=read_dataset_from_files(train_set,features,label,window=window)
X_train=dataset["X"]
y_train=dataset["y"]

l=[]
for i,f in enumerate(test_set):
    print(i,f)
    test_data=read_csv(f,features,label,window=window)
    X_test=test_data["X"]
    y_test=test_data["y"]
    X_train_scaled,X_test_scaled,_=scale(X_train,X_test)
    
    X_train_scaled=X_train_scaled.reshape(-1,window,len(features))
    X_test_scaled=X_test_scaled.reshape(-1,window,len(features))
        
    model=RNN(checkpoint_filepath=f'./logs/model/dgp_rnn_aug_{i}',seed=seed,restore=restore)
    model.train(X_train_scaled,X_test_scaled,y_train,y_test,epochs=epochs,batch_size=batch_size)
    l.append(model.metrics)
    X_train=np.concatenate([X_train,X_test],axis=0)
    annot=model.pred_fn(X_test_scaled)
    y_train=np.concatenate([y_train,annot],axis=0)

df=pd.DataFrame(l)
df.to_csv("./logs/exp/dgp_rnn_aug.csv",index=False)