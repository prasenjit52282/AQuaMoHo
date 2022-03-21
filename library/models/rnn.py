import numpy as np
import tensorflow as tf
from ..metrics import *
from ..constants import seed,features,label,window
from ..utils import read_dataset,read_dataset_from_files
from ..helper import split_scale_dataset,set_scale_dataset
import tensorflow.keras.backend as K

class Attention(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(Attention,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='atten_w',shape=(input_shape[-1],1),initializer='random_normal',trainable=True)
        self.b=self.add_weight(name='atten_b',shape=(input_shape[1],1),initializer='zeros',trainable=True)        
        super(Attention, self).build(input_shape)
 
    def call(self,x):
        e = K.tanh(K.dot(x,self.W)+self.b)
        e = K.squeeze(e, axis=-1)   
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

class RNN:
    """
    Usage:
    
    rnn=RNN({'lstm1':dict(units=32,seq=True),
             'dp1':dict(rate=0.2),
             'atten':dict(),
             'fc':dict(units=5,activ="softmax")})

    rnn.train_on_file_sets(["./Data/Dgp/Device1_merged.csv"],["./Data/Dgp/Device1_merged.csv"],10,32)
    rnn.train_on_files("./Data/Dgp/*",10,32)
    """
    def __init__(self,arch={},checkpoint_filepath="./logs/model/checkpoint",seed=seed):
        tf.random.set_seed(seed)
        self.model=self.get_model(arch)
        self.checkpoint_filepath=checkpoint_filepath
        self.pred_fn=lambda x:np.argmax(self.model.predict(x),axis=1)
        
    def get_model(self,arch={}):
        model=tf.keras.Sequential()
        for layer,conf in arch.items():
            if 'lstm' in layer:
                model.add(tf.keras.layers.LSTM(units=conf["units"],return_sequences=conf["seq"],name=layer))
            if 'dp' in layer:
                model.add(tf.keras.layers.Dropout(rate=conf["rate"],name=layer))
            if 'atten' in layer:
                model.add(Attention(name=layer))
            if 'fc' in layer:
                model.add(tf.keras.layers.Dense(units=conf["units"],activation=conf["activ"],name=layer))
        model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
        return model
            
        
    def train(self,X_train,X_test,y_train,y_test,epochs=100,batch_size=32):
        self.X_train,self.X_test,self.y_train,self.y_test=\
        X_train,X_test,y_train,y_test
        model_checkpoint_callback=\
        tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_filepath,save_weights_only=True,
                                           monitor='val_accuracy',mode='max',save_best_only=True)
        self.history=self.model.fit(self.X_train,self.y_train,epochs=epochs,batch_size=batch_size,
                                    validation_data=(self.X_test,self.y_test),
                                    callbacks=[model_checkpoint_callback])
        self.restore_best_weights()
        self.print_performance()
        
    def restore_best_weights(self):
        print("restoring best weights...")
        self.model.load_weights(self.checkpoint_filepath)
        
    def train_on_dataset(self,dataset,test_size,epochs,batch_size):
        X_train,X_test,y_train,y_test,_=split_scale_dataset(dataset,test_size)
        self.train(X_train,X_test,y_train,y_test,epochs,batch_size)
        
    def train_on_sets(self,train_dataset,test_dataset,epochs,batch_size):
        X_train,X_test,y_train,y_test,_=set_scale_dataset(train_dataset,test_dataset)
        self.train(X_train,X_test,y_train,y_test,epochs,batch_size)
        
    def train_on_files(self,pattern,test_size,epochs,batch_size,features=features,label=label,window=window):
        data=read_dataset(pattern,features,label,window)
        self.train_on_dataset(data,test_size,epochs,batch_size)
        
    def train_on_file_sets(self,train_files,test_files,epochs,batch_size,features=features,label=label,window=window):
        train_data=read_dataset_from_files(train_files,features,label,window)
        test_data=read_dataset_from_files(test_files,features,label,window)
        self.train_on_sets(train_data,test_data,epochs,batch_size)
    
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