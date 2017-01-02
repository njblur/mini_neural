import numpy as np
import neural


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LSTM:
    """
    simple implementation for lstm
    """

    def __init__(self, input_size, hidden_size):
        self.feature_size = input_size + hidden_size
        self.hidden_size = hidden_size
        self.c = np.zeros([hidden_size, 1])
        self.h = np.zeros([hidden_size, 1])

        self.f_weights = np.random.randn(hidden_size, self.feature_size)*0.1
        self.f_bias = np.ones([hidden_size, 1])*3
        self.i_weights = np.random.randn(hidden_size, self.feature_size)*0.1
        self.i_bias = np.ones([hidden_size, 1])*0.001
        self.o_weights = np.random.randn(hidden_size, self.feature_size)*0.1
        self.o_bias = np.ones([hidden_size, 1])*0.001

        self.wct_weights = np.random.randn(hidden_size, self.feature_size)*0.1
        self.wct_bias = np.zeros([hidden_size, 1])
        self.d_c = np.zeros_like(self.c)


    def clear_state(self):
        self.c[:, :] = 0
        self.h[:, :] = 0
        self.d_c[:, :] = 0

    def forward(self, x):
        self.x = x
        self.pre_h = self.h.copy()
        self.pre_c = self.c.copy()
        self.hx = hx = np.vstack([self.pre_h, self.x])

        self.f_gate = np.matmul(self.f_weights, hx)
        self.f_gate += self.f_bias
        self.f_gate = sigmoid(self.f_gate)

        self.i_gate = np.matmul(self.i_weights, hx)
        self.i_gate += self.i_bias
        self.i_gate = sigmoid(self.i_gate)

        self.o_gate = np.matmul(self.o_weights, hx)
        self.o_gate += self.o_bias
        self.o_gate = sigmoid(self.o_gate)

        self.c_tmp = np.matmul(self.wct_weights, hx)
        self.c_tmp += self.wct_bias
        self.c_tmp = np.tanh(self.c_tmp)

        self.c = self.pre_c * self.f_gate + self.i_gate * self.c_tmp
        self.tanh_c = np.tanh(self.c)
        self.h = self.tanh_c * self.o_gate

        return self.h

    def backward(self, dy):
        self.d_hx = np.zeros_like(self.hx)
        self.d_o_gate = dy * self.tanh_c
        self.d_tanh_c = dy * self.o_gate
        self.d_c += self.d_tanh_c * (1 - self.tanh_c**2)

        self.d_pre_c = self.d_c*self.f_gate

        self.d_f_gate = self.d_c * self.pre_c
        self.d_i_gate = self.d_c * self.c_tmp
        self.d_c_tmp = self.d_c * self.i_gate

        self.d_c += self.d_pre_c

        self.d_c_tmp = self.d_c_tmp * (1 - self.c_tmp * self.c_tmp)
        self.d_wct_bias = self.d_c_tmp
        self.d_wct_weights = np.matmul(self.d_c_tmp, self.hx.T)
        self.d_hx += np.matmul(self.wct_weights.T, self.d_c_tmp)

        self.d_o_gate = self.d_o_gate * self.o_gate * (1 - self.o_gate)
        self.d_o_bias = self.d_o_gate
        self.d_o_weights = np.matmul(self.d_o_gate, self.hx.T)
        self.d_hx += np.matmul(self.o_weights.T, self.d_o_gate)

        self.d_i_gate = self.d_i_gate * self.i_gate * (1 - self.i_gate)
        self.d_i_bias = self.d_i_gate
        self.d_i_weights = np.matmul(self.d_i_gate, self.hx.T)
        self.d_hx += np.matmul(self.i_weights.T, self.d_i_gate)

        self.d_f_gate = self.d_f_gate * self.f_gate * (1 - self.f_gate)
        self.d_f_bias = self.d_f_gate
        self.d_f_weights = np.matmul(self.d_f_gate, self.hx.T)
        self.d_hx += np.matmul(self.f_weights.T, self.d_f_gate)

        for w in [self.d_f_bias,self.d_i_bias,self.d_o_bias,self.d_f_weights,self.d_i_weights,self.d_o_weights,self.wct_bias,self.wct_weights]:
            np.clip(w,-8,8,out=w)

        return self.d_hx[self.h.shape[0]:, :]

    def apply(self, learning_rate):
        self.o_bias -= learning_rate * self.d_o_bias
        self.o_weights -= learning_rate * self.d_o_weights

        self.i_bias -= learning_rate * self.d_i_bias
        self.i_weights -= learning_rate * self.d_i_weights

        self.f_bias -= learning_rate * self.d_f_bias
        self.f_weights -= learning_rate * self.d_f_weights

        self.wct_bias -= learning_rate * self.d_wct_bias
        self.wct_weights -= learning_rate * self.d_wct_weights

if __name__ == "__main__":
    """
    test code
    """
    data = open("input.txt").read()
    vocab = list(set(data))
    char_to_idx = {c:i for i,c in enumerate(vocab) }
    idx_to_char = {i:c for i,c in enumerate(vocab) }
    hidden_size = 20
    vocab_size = len(vocab)
    data_size = len(data)
    loop = 5000
    learning_rate = 1.5
    learning_rate_decay = 1.0
    lstm = LSTM(vocab_size, hidden_size)
    l_softmax = neural.layer(hidden_size,vocab_size,"softmax")
    
    for j in xrange(loop):
        lstm.clear_state()
        for i in xrange(data_size-1):
            input_char = data[i]
            target_char = data[i+1]
            input_idx = char_to_idx[input_char]
            target_idx = char_to_idx[target_char]
            input_vec = np.zeros([vocab_size, 1])
            target_vec = np.zeros([vocab_size, 1])
            input_vec[input_idx, 0] = 1
            target_vec[target_idx, 0] = 1
            y1 = lstm.forward(input_vec)
            y2 = l_softmax.forward(y1)
            loss = -target_vec*np.log(y2)
            loss = np.sum(loss)
            dy2 = y2 - target_vec
            dy1 = l_softmax.backward(dy2)
            dy = lstm.backward(dy1)
            lstm.apply(learning_rate)
            l_softmax.apply_gradients(learning_rate)
            # if(i%10 == 0): print loss
        learning_rate *= learning_rate_decay
        print "iter %d finished "%j
        lstm.clear_state()
        start = data[0]
        start_idx = char_to_idx[start]
        start_vec = np.zeros([vocab_size, 1])
        start_vec[start_idx] = 1
        seq = []
        for _ in range(30):
            seq.append(start_idx)
            y = lstm.forward(start_vec)
            start_vec = l_softmax.forward(y)
            # print(start_vec)
            start_idx = np.argmax(start_vec)
            # start_idx = np.random.choice(vocab_size, p=start_vec.ravel())
            start_vec[:,:] = 0
            start_vec[start_idx,0] = 1
        # print(seq)
        txt = ''.join(idx_to_char[ix] for ix in seq)
        print(txt)

