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
        self.pre_h = self.h.copy()
        self.pre_c = self.c.copy()
        self.hx = hx = np.vstack([self.pre_h,self.x])

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
        self.c = self.pre_c*self.f_gate + self.i_gate*self.c_tmp
        self.tanh_c = np.tanh(self.c)
        self.h = self.tanh_c*self.o_gate

        return self.h
    def backward(self,dy):
        self.d_hx = np.zeros_like(self.hx)
        self.d_o_gate = dy * self.tanh_c
        self.d_c = dy*self.o_gate*(1-self.tanh_c*self.tanh_c)
        
        self.d_f_gate = self.d_c*self.pre_c
        self.d_i_gate = self.d_c*self.c_tmp
        self.d_c_tmp = self.d_c*self.i_gate

        self.d_c_tmp = self.d_c_tmp*(1-self.c_tmp*self.c_tmp)
        self.d_wct_bias = self.d_c_tmp
        self.d_wct_weights = np.matmul(self.d_c_tmp,self.hx.T)
        self.d_hx += np.matmul(self.wct_weights.T,self.d_c_tmp)

        self.d_o_gate = self.d_o_gate*self.o_gate*(1-self.o_gate)
        self.d_o_bias = self.d_o_gate
        self.d_o_weights = np.matmul(self.d_o_gate,self.hx.T)
        self.d_hx += np.matmul(self.o_weights.T,self.d_o_gate)

        self.d_i_gate = self.d_i_gate*self.i_gate*(1-self.i_gate)
        self.d_i_bias = self.d_i_gate
        self.d_i_weights = np.matmul(self.d_i_gate,self.hx.T)
        self.d_hx += np.matmul(self.i_weights.T,self.d_i_gate)
        
        self.d_f_gate = self.d_f_gate*self.f_gate*(1-self.f_gate)
        self.d_f_bias = self.d_f_gate
        self.d_f_weights = np.matmul(self.d_f_gate,self.hx.T)
        self.d_hx += np.matmul(self.f_weights.T,self.d_f_gate)

        return self.d_hx[self.h.shape[0]:,:]
    def apply(self,learning_rate):
        self.o_bias -= learning_rate*self.d_o_bias
        self.o_weights -= learning_rate*self.d_o_weights

        self.i_bias -= learning_rate*self.d_i_bias
        self.i_weights -= learning_rate*self.d_i_weights

        self.f_bias -= learning_rate*self.d_f_bias
        self.f_weights -= learning_rate*self.d_f_weights

        self.wct_bias -= learning_rate*self.d_wct_bias
        self.wct_weights -= learning_rate*self.d_wct_weights
       
if __name__ == "__main__":
    lstm = LSTM(5,8)
    lstm.clear_state()
    x = np.random.randint(10,size=[5,1])
    y = lstm.forward(x)
    dx = lstm.backward(y-1)
    print dx



                          
