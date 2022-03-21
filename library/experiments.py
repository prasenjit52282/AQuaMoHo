import glob
import pandas as pd

get_files=lambda city,devices:[f'./Data/{city}/Device_{f}_merged.csv' for f in devices]

def run_experiment_with(model,city,train_devices,test_devices,epochs=None,batch_size=None):
    train_files=get_files(city,train_devices)
    test_files=get_files(city,test_devices)
    model.train_on_file_sets(train_files,test_files,epochs=epochs,batch_size=batch_size)
    perf=model.metrics
    perf["train_dev"]=train_devices
    perf["test_dev"]=test_devices
    return perf

def overall_experiments(exp_name,model,city,epochs=None,batch_size=None):
    devices=sorted([int(f.split("_")[1]) for f in glob.glob(f"./Data/{city}/*")])
    overall_exp={}
    for dev in devices:
        overall_exp[dev]=dict(train_devices=list(set(devices)-set([dev])),test_devices=[dev])
    stat=[]
    for exp, conf in overall_exp.items():
        met=run_experiment_with(model,city,conf["train_devices"],conf["test_devices"],epochs,batch_size)
        stat.append(met)
    df=pd.DataFrame(stat)
    df.to_csv(f"./logs/exp/{exp_name}_{city}.csv",index=False)
    return df