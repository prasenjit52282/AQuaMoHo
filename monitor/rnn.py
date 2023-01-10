import sys
sys.path.append("../")
from library.models.rnn import RNN

model_fn=lambda path='./logs/model/checkpoint',restore=False: RNN(checkpoint_filepath=path,restore=restore)
model=model_fn(path='../logs/model/Dgp_split_rnn',restore=True)
fig=model.train_on_files(f"../Data/Dgp/*",test_size=0.3,epochs=1,batch_size=32)
print("Running Inferences...")

while True:
    model.pred_fn(model.X_test[:1,:])