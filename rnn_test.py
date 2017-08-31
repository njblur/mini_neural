import numpy as np 
import rnn
import neural
import IPython
"""
use rnn to judge if a string has a valid quotation marks, a even number of quotation marks are treated valid
"abc" "a"b "aab" are valid, while abc "abc abc" "a"" are invalids, this would verify the 'memory' ability of rnn
"""
chars = 'abcd"'
charset = set(chars)
char_num = len(charset)
char2idx = {c:i for i,c in enumerate(charset)}
idx2char = {i:c for i,c in enumerate(charset)}
allchars = np.array(list(charset))
print allchars
def data_to_str(t):
    ids = np.argmax(t,axis=1).reshape(seq_len)
    chars = [idx2char[i] for i in ids]
    strs = ''.join(chars)
    return strs

train_size = 5000
test_size = 100

seq_len = 6

rnn_layer = rnn.rnn(char_num,50,10)
fc = neural.layer(10,1,'sigmoid')

train_data = np.zeros(shape=(train_size+test_size,seq_len,char_num))
train_label = np.zeros(shape=(train_size+test_size,1))

for i in range(train_size+test_size):
    seqids = np.random.randint(0,char_num,size=seq_len)
    seqs = allchars[seqids]
    qts=np.count_nonzero(seqs=='"')
    if qts != 0 and qts %2 == 0:
        train_label[i] = 1
    for j in range(seq_len):
        idx = char2idx[seqs[j]]
        train_data[i,j,idx] = 1
test_data = train_data[-test_size:]
test_label = train_label[-test_size:]

train_data = train_data[:train_size]
train_label = train_label[:train_size]
epoch = 30
learning_rate = 0.1
for j in range(epoch):
    for i in range(train_size):
        x = train_data[i].reshape(seq_len,char_num,1)
        l = train_label[i]
        y,h = rnn_layer.forward(x)
        dy,dh = np.zeros_like(y),np.zeros_like(h)
        o = fc.forward(y[-1])
        loss = neural.sigmoid_loss(o,l)
        dldo = (1-l)/(1-o) - l/o
        dy[-1] = fc.backward(dldo)
        dx = rnn_layer.backward(dy,dh)
        fc.apply_gradients(learning_rate)
        rnn_layer.apply_gradients(learning_rate)
        if i%20 == 0 :
            print loss
    learning_rate *= 0.9

total_loss = 0.0
correct = 0
for i in range(test_size):
    x = test_data[i].reshape(seq_len,char_num,1)
    l = test_label[i]
    y,h = rnn_layer.forward(x)
    dy,dh = np.zeros_like(y),np.zeros_like(h)
    o = fc.forward(y[-1])
    loss = neural.sigmoid_loss(o,l)
    text = data_to_str(x)
    if(loss < 0.35):
        correct += 1
        print "correct for " + text
    else:
        print "wrong for " + text
    total_loss += loss
total_loss /= test_size
correct_rate = correct*1.0/test_size

print "test loss is " + str(total_loss)
print "correct rate is " + str(correct_rate)


IPython.embed()