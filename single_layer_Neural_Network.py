import numpy as np
import matplotlib.pyplot as plt
x=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
d=np.array([[1,0],[1,0],[0,1],[0,1]])
w=np.random.rand(d.shape[1],x.shape[1])
learning_rate=1e-2
N=400
E_plot=np.zeros((N))
for step in range(N):
    NET=np.dot(x,w.T)
    y=np.where(NET>=1,1,0)
    E=(d-y)**2
    dw=learning_rate*np.dot(E.T,x)
    w+=dw
    if step%100==0:
        print(y)
    E_plot[step]=E.sum()
plt.plot(E_plot)
plt.show()
