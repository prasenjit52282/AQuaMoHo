import glob
import numpy as np
import pandas as pd
from library.constants import *
from library.models.rnn import RNN
from library.models.rf import RandomForest

#-----------------------------------------------------------------
#Delhi
num_random_exp=10
dev_list=glob.glob("./Data/Delhi/*")

def get_train_test_lists(num_test_dev=1):
    test_devs=np.random.choice(dev_list,num_test_dev,replace=False)
    train_devs=np.array(list(set(dev_list)-set(test_devs)))
    return train_devs,test_devs

#Random Forest
l=[]
np.random.seed(seed)
for num_test_dev in [1,2,3,4,5,6]:
    for random_exp in range(num_random_exp):
        train_devs,test_devs=get_train_test_lists(num_test_dev)
        model=RandomForest()
        model.train_on_file_sets(train_devs,test_devs)
        met=model.metrics
        met["num_test_dev"]=num_test_dev
        met["rand_exp_id"]=random_exp
        l.append(met)
df=pd.DataFrame(l)
df.to_csv("./logs/exp/delhi_rf_varryDiv.csv",index=False)

#RNN
l=[]
np.random.seed(seed)
for num_test_dev in [1,2,3,4,5,6]:
    for random_exp in range(num_random_exp):
        train_devs,test_devs=get_train_test_lists(num_test_dev)
        model=RNN(checkpoint_filepath=f'./logs/model/delhi_rnn_vd',seed=seed)
        model.train_on_file_sets(train_devs,test_devs,epochs=epochs,batch_size=batch_size)
        met=model.metrics
        met["num_test_dev"]=num_test_dev
        met["rand_exp_id"]=random_exp
        l.append(met)
df=pd.DataFrame(l)
df.to_csv("./logs/exp/delhi_rnn_varryDiv.csv",index=False)