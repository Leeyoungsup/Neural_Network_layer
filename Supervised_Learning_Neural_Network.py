import numpy as np


def numerical_derivative(f, x):
    delta_x = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x)
        x[idx] = float(tmp_val) - delta_x
        fx2 = f(x)
        grad[idx] = (fx1 - fx2) / (2 * delta_x)
        x[idx] = tmp_val
        it.iternext()
    return grad


def loss_func(x, t):
    delta = 1e-7
    z = np.dot(x, W) + b
    y = sigmoid(z)
    return -np.sum(t * np.log(y + delta) + (1 - t) * np.log(1 - y + delta))


def predict(x):
    z = np.dot(x, W) + b
    y = sigmoid(z)
    result=np.where(y>=0.5,1,0)
    return y, result


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x_data = np.array([[1,1,0,0]])
t_data = np.array([[1,0]])
W = np.random.rand(x_data.shape[1], t_data.shape[1])
b = np.random.rand(1)
f = lambda x: loss_func(x_data, t_data)

learning_rate = 1e-2
for step in range(40001):
    W -= learning_rate * numerical_derivative(f, W)
    b -= learning_rate * numerical_derivative(f, b)
    if (step % 400 == 0):
        print("step=", step, "error value=", loss_func(x_data, t_data), "W=", W, "b=", b,)
for i in range(x_data.shape[0]):
    valueConfirm = predict(x_data[i])
    print(valueConfirm)
