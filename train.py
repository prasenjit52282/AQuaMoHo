import argparse
from library.models.rnn import RNN
from library.constants import epochs,batch_size
from library.models.rf import RandomForest,OldRandomForest
from library.experiments import experiment,datasplit_experiment

parser = argparse.ArgumentParser(description='Training for different experiments with rf and rnn')
parser.add_argument('--city', type=str, default="Dgp", help='Required name of city (Dgp/Delhi)')
parser.add_argument('--restore', help='Required decision to restore weights',action='store_true')

args = parser.parse_args()
city=args.city
restore=args.restore
print(f"Training on city: {city} with restore: {restore}")

#Model function
old_rf_model_fn=lambda path=None,restore=False: OldRandomForest()
rf_model_fn=lambda path=None,restore=False: RandomForest()
rnn_model_fn=lambda path='./logs/model/checkpoint',restore=False: RNN(checkpoint_filepath=path,restore=restore)


#Old Random Forest
df=datasplit_experiment("split_rf_old",city,old_rf_model_fn,test_size=0.3,epochs=epochs,batch_size=batch_size,restore=restore)
df=experiment('overall_rf_old',city,old_rf_model_fn,restore=restore)
df=experiment('sim_rf_old',city,old_rf_model_fn,restore=restore)
df=experiment('dis_rf_old',city,old_rf_model_fn,restore=restore)


#Random Forest
df=datasplit_experiment("split_rf",city,rf_model_fn,test_size=0.3,epochs=epochs,batch_size=batch_size,restore=restore)
df=experiment('overall_rf',city,rf_model_fn,restore=restore)
df=experiment('sim_rf',city,rf_model_fn,restore=restore)
df=experiment('dis_rf',city,rf_model_fn,restore=restore)


#RNN
df=datasplit_experiment("split_rnn",city,rnn_model_fn,test_size=0.3,epochs=epochs,batch_size=batch_size,restore=restore)
df=experiment('overall_rnn',city,rnn_model_fn,epochs=epochs,batch_size=batch_size,restore=restore)
df=experiment('sim_rnn',city,rnn_model_fn,epochs=epochs,batch_size=batch_size,restore=restore)
df=experiment('dis_rnn',city,rnn_model_fn,epochs=epochs,batch_size=batch_size,restore=restore)