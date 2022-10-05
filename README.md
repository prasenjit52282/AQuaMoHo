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

5.(Varry Window size experiments from ws param)
* python3 varryWindow.py --city="Dgp" --ws="6,12,18,24,30"
* python3 varryWindow.py --city="Delhi" --ws="6,12,18,24,30"

6.(Training for Both City 70-30 split)
* python3 train.py --city="D*"
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

5.(Varry Window size experiments from ws param)
* python3 varryWindow.py --city="Dgp" --ws="6,12,18,24,30" --restore
* python3 varryWindow.py --city="Delhi" --ws="6,12,18,24,30" --restore

6.(Training for Both City 70-30 split)
* python3 train.py --city="D*" --restore

<hr>

Plotting
--------
* run plot.ipynb file