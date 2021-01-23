# ODE4RNN
ODE-Solver for Recurrent Neural Networks in MATLAB

This ODE-Solver can be used for simulating recurrent neural networks (RNNs) modeling the dynamics function `f(x,u,t) = dxdt` of arbitrary dynamical systems.
The neural network takes sequences of driver data D as input data and gives back a sequence of output data DXDT.
The driver sequences D consist of a state sequence X and a control input sequence U.

## How to use ODE4RNN
To simulate a recurrent neural network you call ode4rnn with the following arguments:
```
[T, D] = ode4rnn(predictFcn, controlFcn, tSpan, stepSize4Rnn, x_0, maxSequenceLength, UsePlotting)
```
```DXDT = predictFcn(D)``` - function that takes sequence of driver data D and gives back the corresponding output sequence DXDT

```U = controlFcn(T)``` - function that takes sequence of time stamps and gives back the corresponding control vectors

```tSpan = [t_0, t_end]``` - vector of start and end time stamps to simulate for

```stepSize4Rnn``` - positive double number defining the time step size of the sequence data for the recurrent neural network

```x_0``` - initial state

```maxSequenceLength``` - positive integer defining the max. sequence data length for the recurrent neural network (the lower the faster)

```UsePlotting``` - (optional) plotting the inherent interpolation
