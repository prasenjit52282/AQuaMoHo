from ..metrics import *
from ..constants import seed,features,label
from ..utils import read_dataset,read_dataset_from_files
from ..helper import split_scale_dataset,set_scale_dataset
from sklearn.ensemble import RandomForestClassifier

class RandomForest:
    def __init__(self,n_est=100,max_depth=20):
        self.model=RandomForestClassifier(n_estimators=n_est,max_depth=max_depth,random_state=seed)
        self.pred_fn=lambda x:self.model.predict(x)
        
    def train(self,X_train,X_test,y_train,y_test,epochs=None,batch_size=None):
        self.X_train,self.X_test,self.y_train,self.y_test=\
        X_train,X_test,y_train,y_test
        self.model.fit(self.X_train,self.y_train)
        self.print_performance()
        
    def train_on_dataset(self,dataset,test_size,epochs=None,batch_size=None):
        X_train,X_test,y_train,y_test,_=split_scale_dataset(dataset,test_size)
        self.train(X_train,X_test,y_train,y_test)
        
    def train_on_sets(self,train_dataset,test_dataset,epochs=None,batch_size=None):
        X_train,X_test,y_train,y_test,_=set_scale_dataset(train_dataset,test_dataset)
        self.train(X_train,X_test,y_train,y_test)
        
    def train_on_files(self,pattern,test_size,features=features,label=label,epochs=None,batch_size=None):
        data=read_dataset(pattern,features,label)
        self.train_on_dataset(data,test_size)
        
    def train_on_file_sets(self,train_files,test_files,features=features,label=label,epochs=None,batch_size=None):
        train_data=read_dataset_from_files(train_files,features,label)
        test_data=read_dataset_from_files(test_files,features,label)
        self.train_on_sets(train_data,test_data)
    
    def print_performance(self):
        res="Train............\n"
        res+=str(get_performance(self.X_train,self.y_train,self.pred_fn))+"\n"
        res+=get_report(self.X_train,self.y_train,self.pred_fn)
        res+=str(get_confusion_matrix(self.X_train,self.y_train,self.pred_fn))+"\n"
        res+="\nTest............\n"
        res+=str(get_performance(self.X_test,self.y_test,self.pred_fn))+"\n"
        res+=get_report(self.X_test,self.y_test,self.pred_fn)
        res+=str(get_confusion_matrix(self.X_test,self.y_test,self.pred_fn))+"\n"
        self.evaluation=res
        print(res)
    
    @property
    def metrics(self):
        met={}
        train_metrics=get_performance(self.X_train,self.y_train,self.pred_fn)
        test_metrics=get_performance(self.X_test,self.y_test,self.pred_fn)
        for m,v in train_metrics.items():
            met[m+'_train']=v
        for m,v in test_metrics.items():
            met[m+'_test']=v
        return met