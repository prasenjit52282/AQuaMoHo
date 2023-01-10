import sys
sys.path.append("../")
from library.models.rf import OldRandomForest

model_fn=lambda path=None,restore=False: OldRandomForest()
model=model_fn(path='none',restore=True)
fig=model.train_on_files(f"../Data/Dgp/*",test_size=0.3,epochs=None,batch_size=32)
print("Running Inferences...")

while True:
    model.pred_fn(model.X_test[:1,:])