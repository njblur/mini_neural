import numpy as np
class rnn:
    def __init__(self,input_size,hidden_size,out_size):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.out_size = out_size
        self.Wx = np.random.normal(loc=0.0,scale=1/np.sqrt(input_size*hidden_size),size=[hidden_size,input_size])
        self.Bx = np.zeros(shape=[hidden_size,1])

        self.Wh = np.random.normal(loc=0.0,scale=1/np.sqrt(hidden_size*hidden_size),size=[hidden_size,hidden_size])

        self.Wy = np.random.normal(loc=0.0,scale=1/np.sqrt(out_size*hidden_size),size=[out_size,hidden_size])
        self.By = np.zeros(shape=[out_size,1])
    def forward(self,x):
        self.seq_size = len(x)
        self.x = x
        self.h = np.zeros(shape=(self.seq_size+1,self.hidden_size,1))
        self.hz = np.zeros_like(self.h)
        self.y = np.zeros(shape=(self.seq_size,self.out_size,1))
        for t in range(self.seq_size):
            self.hz[t] = self.Wx.dot(x[t]) + self.Wh.dot(self.h[t-1]) + self.Bx
            self.h[t] = np.tanh(self.hz[t])
            self.y[t] = self.Wy.dot(self.h[t]) + self.By #it would be some waste if only some of the Ys are needed
        return self.y,self.h # last item in self.h are zeros for temp use,h is returned since some models like encoder-decoder attention need it.
    def backward(self,dy,dh):
        self.dWx = np.zeros_like(self.Wx)
        self.dBx = np.zeros_like(self.Bx)
        self.dWh = np.zeros_like(self.Wh)
        self.dWy = np.zeros_like(self.Wy)
        self.dBy = np.zeros_like(self.By)
        dx = np.zeros_like(self.x)
        for t in reversed(range(self.seq_size)):
            dh[t] += self.Wy.T.dot(dy[t])
            self.dWy += dy[t].dot(self.h[t].T)
            self.dBy += dy[t]

            dzt = dh[t]*(1-self.h[t]**2)
            dx[t] = self.Wx.T.dot(dzt)
            self.dWx = dzt.dot(self.x[t].T)
            dh[t-1] += self.Wh.T.dot(dzt)
            self.dWh += dzt.dot(self.h[t-1].T)
            self.dBx += dzt
        return dx
    def apply_gradients(self,learning_rate):
        self.Wx -= self.dWx*learning_rate
        self.Bx -= self.dBx*learning_rate
        self.Wh -= self.dWh*learning_rate
        self.Wy -= self.dWy*learning_rate
        self.By -= self.dBy*learning_rate
if __name__ == '__main__':
    t = rnn(8,5,10)
    x = np.random.randint(0,5,size=[5,8,1]).astype('float16')
    y,h = t.forward(x)
    dy,dh=np.zeros_like(y),np.zeros_like(h)
    dy[-1] = 0.5
    dy[-2] = 0.2
    dy[-3] = .07
    dx = t.backward(dy,dh)
    t.apply_gradients(0.01)
 
    print dx
    




        


        
