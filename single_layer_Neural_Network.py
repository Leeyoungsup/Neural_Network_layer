import numpy as np
def sig(x):
    y=1/(1+np.exp(-x))
    y=np.where(y>=0.5,1,0)
    return y
x=np.array([[1,0,0,0]])
d=np.array([[1,0]])
w=np.random.rand(d.shape[1],x.shape[1])
learning_rate=1e-4
NET=np.dot(x,w.T)
y=sig(NET)
E=d-y

print(y)