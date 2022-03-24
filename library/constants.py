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

label='AQI'

window=18

seed=42


model_arch={'lstm1':dict(units=32,seq=True,l2=None),
            'atten':dict(),
            'dp1':dict(rate=0.2),
            'fc1':dict(units=32,activ="tanh",l2=None),
            'dp2':dict(rate=0.2),
            'fc2':dict(units=5,activ="softmax",l2=None)}

epochs=1000
batch_size=64