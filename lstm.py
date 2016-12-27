import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LSTM:
    """
    simple implementation for lstm
    """
    def __init__(self,input_size,hidden_size):
        self.feature_size = input_size + hidden_size
        self.hidden_size = hidden_size
        self.c = np.zeros([hidden_size,1])
        self.h = np.zeros([hidden_size,1])

        self.f_weights = np.random.randn(hidden_size,self.feature_size)
        self.f_bias = np.random.randn(hidden_size,1)
        self.i_weights = np.random.randn(hidden_size,self.feature_size)
        self.i_bias = np.random.randn(hidden_size,1)        
        self.o_weights = np.random.randn(hidden_size,self.feature_size)
        self.o_bias = np.random.randn(hidden_size,1)
        
        self.wct_weights = np.random.randn(hidden_size,self.feature_size)
        self.wct_bias = np.random.randn(hidden_size,1)
    
    def clear_state(self):
        self.c = np.zeros([self.hidden_size,1])
        self.h = np.zeros([self.hidden_size,1])
    def forward(self,x):
        self.x = x
        hx = np.vstack([self.h,x])

        self.f_gate = np.matmul(self.f_weights,hx)
        self.f_gate += self.f_bias
        self.f_gate = sigmoid(self.f_gate)

        self.i_gate = np.matmul(self.i_weights,hx)
        self.i_gate += self.i_bias
        self.i_gate = sigmoid(self.i_gate)

        self.o_gate = np.matmul(self.o_weights,hx)
        self.o_gate += self.o_bias
        self.o_gate = sigmoid(self.o_gate)
        
        self.c_tmp = np.matmul(self.wct_weights,hx)
        self.c_tmp += self.wct_bias
        self.c_tmp = np.tanh(self.c_tmp)
        self.c = self.c*self.f_gate + self.i_gate*self.c_tmp
        self.h = np.tanh(self.c)*self.o_gate

        return self.h
        
        
if __name__ == "__main__":
    lstm = LSTM(5,8)
    lstm.clear_state()
    x = np.random.randint(10,size=[5,1])
    y = lstm.forward(x)
    print y

    print sigmoid(-10)

                          
