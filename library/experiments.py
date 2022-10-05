import glob
import pandas as pd

get_files=lambda city,devices:[f'./Data/{city}/Device_{f}_merged.csv' for f in devices]

def run_experiment_with(model,city,train_devices,test_devices,epochs=None,batch_size=None):
    train_files=get_files(city,train_devices)
    test_files=get_files(city,test_devices)
    auc_roc_fig=model.train_on_file_sets(train_files,test_files,epochs=epochs,batch_size=batch_size)
    perf=model.metrics
    perf["train_dev"]=train_devices
    perf["test_dev"]=test_devices
    return perf,auc_roc_fig

def read_overall_confs(city):
    exps={}
    devices=sorted([int(f.split("_")[1]) for f in glob.glob(f"./Data/{city}/*")])
    for dev in devices:
        exps[dev]=dict(train_devices=list(set(devices)-set([dev])),test_devices=[dev])
    return exps

def read_similarity_confs(city,similarity):
    exps={}
    temp=dict(pd.read_csv(f"./Data/metadata/{city}_{similarity}.csv").T)
    for e,conf in temp.items():
        exps[e]=dict(train_devices=[conf["train_devices"]],test_devices=[conf["test_devices"]])
    return exps

def get_experiment_config(exp_name,city):
    if 'overall' in exp_name:
        return read_overall_confs(city)
    elif 'sim' in exp_name:
        return read_similarity_confs(city,'sim')
    elif 'dis' in exp_name:
        return read_similarity_confs(city,'dis')

def experiment(exp_name,city,model_fn,epochs=None,batch_size=None,restore=False):
    """
    exp_name="(overall/sim/dis)_*"
    city=("Delhi"/"Dgp")
    """
    stat=[]
    nCity=("Both" if city=='*' else city)
    exp_config=get_experiment_config(exp_name,city)
    for exp, conf in exp_config.items():
        save_pattern=nCity+"_"+str(exp)+"_"+exp_name
        met,fig=run_experiment_with(model_fn(path='./logs/model/'+save_pattern,restore=restore),
                                    city,conf["train_devices"],conf["test_devices"],epochs,batch_size)
        fig.savefig('./logs/figs/'+save_pattern+"_auc_roc.png")
        stat.append(met)
    df=pd.DataFrame(stat)
    df.to_csv(f"./logs/exp/{exp_name}_{nCity}.csv",index=False)
    return df

def datasplit_experiment(exp_name,city,model_fn,test_size=0.3,epochs=None,batch_size=None,restore=False):
    nCity=("Both" if city=='*' else city)
    save_pattern=nCity+"_"+exp_name
    model=model_fn(path='./logs/model/'+save_pattern,restore=restore)
    fig=model.train_on_files(f"./Data/{city}/*",test_size=test_size,epochs=epochs,batch_size=batch_size)
    met=model.metrics
    met["train"]=(1-test_size)
    met["test"]=test_size
    df=pd.DataFrame([met])
    df.to_csv(f"./logs/exp/{exp_name}_{nCity}.csv",index=False)
    fig.savefig('./logs/figs/'+save_pattern+"_auc_roc.png")
    return df