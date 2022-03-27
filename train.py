from library.models.rnn import RNN

city="Delhi"
data_pattern=f"./Data/{city}/*"

rnn_model_fn=lambda : RNN({'lstm1':dict(units=128,seq=True,l2=None),
                           'atten':dict(),
                           'dp1':dict(rate=0.2),
                           'fc1':dict(units=128,activ="tanh",l2=0.001),
                           'dp2':dict(rate=0.2),
                           'fc2':dict(units=5,activ="softmax",l2=None)})

rnn=rnn_model_fn()
rnn.train_on_files(data_pattern,test_size=0.3,epochs=200,batch_size=256)
rnn.model.summary()
print(rnn.metrics)