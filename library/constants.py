#default constants
#------------------------
features=\
[
'Temperature',
'Humidity',
'feels_like',
'temp_min',
'temp_max',
'pressure',
'wind_speed',
'wind_deg',
'rain',
'clouds_all',
'weather',
'hour',
'timecluster',
'month',
'season',
'dayofweek'
]

old_features=\
[
'Temperature',
'Humidity',
'feels_like',
'temp_min',
'temp_max',
'pressure',
'wind_speed',
'wind_deg',
'rain',
'clouds_all',
'weather'
]

label='AQI'

window=18

seed=42


model_arch={'lstm1':dict(units=128,seq=True,l2=None),
            'atten':dict(),
            'dp1':dict(rate=0.2),
            'fc1':dict(units=128,activ="tanh",l2=0.001),
            'dp2':dict(rate=0.2),
            'fc2':dict(units=5,activ="softmax",l2=None)}

epochs=200
batch_size=256

figsize=(11,7)
linewidth=4
fontsize=22