from .constants import seed
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def scale(X_train,X_test):
    scaler=MinMaxScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    return X_train,X_test,scaler

def split_scale_dataset(data,test_size,seed=seed):
    X_train, X_test, y_train, y_test = train_test_split(data["X"], data["y"],test_size=test_size,random_state=seed,stratify=data["y"])
    X_train,X_test,scaler=scale(X_train,X_test)
    if 'window_size'in data:
        X_train=X_train.reshape(-1,data["window_size"],data["feat_size"])
        X_test=X_test.reshape(-1,data["window_size"],data["feat_size"])
    return X_train,X_test,y_train,y_test,scaler

def set_scale_dataset(train_data,test_data):
    X_train,X_test,y_train,y_test=train_data["X"],test_data["X"],train_data["y"],test_data["y"]
    X_train,X_test,scaler=scale(X_train,X_test)
    if ('window_size'in train_data) and ('window_size' in test_data):
        X_train=X_train.reshape(-1,train_data["window_size"],train_data["feat_size"])
        X_test=X_test.reshape(-1,test_data["window_size"],test_data["feat_size"])
    return X_train,X_test,y_train,y_test,scaler