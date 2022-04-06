import argparse
import pandas as pd
from library.models.rnn import RNN
from library.constants import epochs,batch_size

parser = argparse.ArgumentParser(description='Training for different experiments with rf and rnn')
parser.add_argument('--city', type=str, default="Dgp", help='Required name of city (Dgp/Delhi)')
parser.add_argument('--ws', type=str, default="6,12,18,24,30", help="Required the list of window sizes")
parser.add_argument('--restore', help='Required decision to restore weights',action='store_true')
args = parser.parse_args()

city=args.city
restore=args.restore
ws_list=[int(e) for e in args.ws.split(",")]
print("config:",city,restore,ws_list)

rnn_model_fn=lambda path='./logs/model/checkpoint',restore=False: RNN(checkpoint_filepath=path,restore=restore)

def datasplit_experiment(exp_name,city,model_fn,test_size=0.3,epochs=None,batch_size=None,window=None,restore=False):
    save_pattern=city+"_"+exp_name
    model=model_fn(path='./logs/model/'+save_pattern,restore=restore)
    model.train_on_files(f"./Data/{city}/*",test_size=test_size,epochs=epochs,batch_size=batch_size,window=window)
    met=model.metrics
    met["train"]=(1-test_size)
    met["test"]=test_size
    return met

l=[]
for ws in ws_list:
    met=datasplit_experiment("window_size",city,rnn_model_fn,0.3,1,batch_size,ws,restore)
    l.append(met)

df=pd.DataFrame(l)
df.to_csv(f"./logs/exp/{city}_window_size.csv",index=False)
