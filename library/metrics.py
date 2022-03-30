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
     'rec_score':recall_score(y,pred,average='weighted')}
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
    else:
        y_score=model.model.predict(model.X_test)
    y_test=tf.keras.utils.to_categorical(model.y_test,n_class)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_class):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fig=plt.figure(figsize=figsize)
    ax=fig.add_subplot(111)
    colors=sns.color_palette('deep',n_colors=n_class)
    
    for i, color in zip(range(n_class), colors):
        ax.plot(fpr[i],tpr[i],color=color,lw=linewidth,label=f"ROC of AQI {i+1} (area = {roc_auc[i]:0.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=linewidth)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=fontsize,fontweight="bold")
    plt.yticks(fontsize=fontsize,fontweight="bold")
    plt.xlabel("False Positive Rate",fontsize=fontsize,fontweight="bold")
    plt.ylabel("True Positive Rate",fontsize=fontsize,fontweight="bold")
    plt.legend(loc="lower right",fontsize=fontsize)
    plt.grid()
    plt.tight_layout()
    plt.close()
    return fig