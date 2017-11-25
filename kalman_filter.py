import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

f = KalmanFilter (dim_x=2, dim_z=1)
f.x = np.array([[2.],    # position
                [0.]])   # velocity
f.x = np.array([2., 0.])
f.F = np.array([[1.,1.],
                [0.,1.]])
f.H = np.array([[1.,0.]])
f.P *= 1000.
f.P = np.array([[1000.,    0.],
                [   0., 1000.] ])
f.R = 5
f.R = np.array([[5.]])
f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)

while True:
    f.predict()
    f.update(get_some_measurement())

    # do something with the output
    x = f.x
    do_something_amazing(x)