from sklearn.metrics import f1_score,precision_score,recall_score,classification_report,confusion_matrix

def get_performance(X,y,pred_fn=None):
    pred=pred_fn(X)
    pref=\
    {'f1_score':f1_score(y,pred,average="weighted"),
     'prec_score':precision_score(y,pred,average="weighted"),
     'rec_score':recall_score(y,pred,average='weighted')}
    return pref

def get_report(X,y,pred_fn=None):
    pred=pred_fn(X)
    return (classification_report(y,pred))
    
    
def get_confusion_matrix(X,y,pred_fn=None):
    pred=pred_fn(X)
    return confusion_matrix(y,pred)