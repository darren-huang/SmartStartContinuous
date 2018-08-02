# pylqr
An implementation of iLQR for trajectory synthesis and control. Use finite difference to approximate gradients and hessians if they are not provided. Also support automatic differentiation with numpy from [autograd](https://github.com/HIPS/autograd). Include an inverted pendulum example as the test case.

Dependencies:

Numpy

Matplotlib (Only for the test)

[autograd](https://github.com/HIPS/autograd) (Only for automatic differentiation)
