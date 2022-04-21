import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from .constants import figsize,linewidth,fontsize
from sklearn.metrics import f1_score,precision_score,recall_score,classification_report,confusion_matrix,roc_curve, auc

def get_performance(X,y,pred_fn=None):
    pred=pred_fn(X)
    pref=\
    {'f1_score':f1_score(y,pred,average="weighted"),
     'prec_score':precision_score(y,pred,average="weighted"),
     'rec_score':recall_score(y,pred,average='weighted'),
     'spec_score':specificity(y,pred)}
    return pref

def get_report(X,y,pred_fn=None):
    pred=pred_fn(X)
    return (classification_report(y,pred))
    
    
def get_confusion_matrix(X,y,pred_fn=None):
    pred=pred_fn(X)
    return confusion_matrix(y,pred)

def AUC_ROC(model,n_class=5):
    if hasattr(model.model,'predict_proba'):
        y_score=model.model.predict_proba(model.X_test)
        examples,out_dim=y_score.shape
        extra=n_class-out_dim
        if extra>0:
            y_score=np.concatenate([y_score,
            np.zeros((examples,extra))],axis=1)
    else:
        y_score=model.model.predict(model.X_test)
    y_test=tf.keras.utils.to_categorical(model.y_test,n_class)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_class):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fig=plt.figure(figsize=(7,5))
    ax=fig.add_subplot(111)
    colors=sns.color_palette('deep',n_colors=n_class)
    
    for i, color in zip(range(n_class), colors):
        ax.plot(fpr[i],tpr[i],color=color,lw=3,label=f"ROC of AQI {i+1} (area = {roc_auc[i]:0.2f})")

    #ax.plot([0, 1], [0, 1], "k--", lw=3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("False Positive Rate",fontsize=18)
    plt.ylabel("True Positive Rate",fontsize=18)
    plt.legend(loc="lower right",fontsize=18)
    plt.grid()
    plt.tight_layout()
    plt.close()
    return fig


def specificity(y_true, y_pred, classes=None):
    if classes is None: # Determine classes from the values
        classes = set(np.concatenate((np.unique(y_true), np.unique(y_pred))))
    specs = []
    for cls in classes:
        y_true_cls = (y_true == cls).astype(int)
        y_pred_cls = (y_pred == cls).astype(int)

        fp = sum(y_pred_cls[y_true_cls != 1])
        tn = sum(y_pred_cls[y_true_cls == 0] == False)

        specificity_val = tn / (tn + fp)
        specs.append(specificity_val)

    return np.mean(specs)