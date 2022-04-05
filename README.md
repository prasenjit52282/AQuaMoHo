Training
---------
1.(split/overall/sim/dis)-rf \& rnn
* python3 train.py --city="Dgp"
* python3 train.py --city="Delhi"

2.(Augument exp- annotate and add train data)
* python3 augument.py

3.(OneByOne exp- add monthly data one by one)
* python3 oneByone.py

4.(varry device exp- varry number of test devices[1-6]-C2)
* python3 varryDiv.py

<hr>

Restoring
---------
1.(split/overall/sim/dis)-rf \& rnn
* python3 train.py --city="Dgp" --restore
* python3 train.py --city="Delhi" --restore

2.(Augument exp- annotate and add train data)
* python3 augument.py --restore

3.(OneByOne exp- add monthly data one by one)
* python3 oneByone.py --restore

4.(varry device exp- varry number of test devices[1-6]-C2)
* NA(python3 varryDiv.py)

<hr>

Plotting
--------
* run plot.ipynb file