import tensorflow as tf
import numpy as np
data = open("input.txt").read()
vocab = list(set(data))
char_to_idx = {c:i for i,c in enumerate(vocab) }
idx_to_char = {i:c for i,c in enumerate(vocab) }
hidden_size = 50
vocab_size = len(vocab)
data_size = len(data)

learning_rate = 0.1
epoch = 20

inputs = tf.placeholder(shape=[1,vocab_size], dtype=tf.float32)
targets = tf.placeholder(shape=[1,vocab_size], dtype=tf.float32)
cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size,forget_bias=1.0)
softmax_weights = tf.Variable(tf.random_normal([vocab_size,hidden_size],dtype=tf.float32,stddev=0.1))
softmax_bias = tf.Variable(tf.zeros([vocab_size,1]))
init_state = cell.zero_state(1,tf.float32)
state = init_state

h,state = cell(inputs,state)

out = tf.matmul(softmax_weights,h,transpose_b=True) +softmax_bias
out_max = tf.nn.softmax(out,dim=0)
out_index = tf.argmax(out_max,0)
loss = tf.nn.softmax_cross_entropy_with_logits(tf.reshape(out,[1,vocab_size]),targets)

loss = tf.reduce_mean(loss)

train = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

in_vec = np.zeros([1,vocab_size])
out_vec = np.zeros([1,vocab_size])

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for i in range(epoch):
        nstate = sess.run(init_state)
        total_loss = 0
        for j in range(data_size-1):
            in_char = data[j]
            in_idx = char_to_idx[in_char]
            out_char = data[j+1]
            out_idx = char_to_idx[out_char]
            in_vec[:,:] = 0
            in_vec[0,in_idx] = 1.0
            out_vec[:,:] = 0
            out_vec[0,out_idx] = 1.0
            nstate,l,t,m,idx = sess.run([state,loss,train,out,out_index],feed_dict={inputs:in_vec,targets:out_vec,init_state:nstate})
            # nstate,t = sess.run([state,train],feed_dict={inputs:in_vec,targets:out_vec,init_state:nstate})
            # print("in {} out {}".format(in_vec,out_vec))
            # print nstate
            # print out_vec
            # print m
            # print l
            # print idx
            total_loss += l
        print("{} loss after {} iter ".format(total_loss,i))

    start = data[0]
    start_idx = char_to_idx[start]
    start_vec = np.zeros([vocab_size, 1])
    start_vec[start_idx] = 1
    seq = []
    nstate = sess.run(init_state)
    for _ in range(10):
        seq.append(start_idx)
        y,nstate = sess.run([out_index,state],feed_dict={inputs:start_vec.T,init_state:nstate})
        start_idx = y[0]
        start_vec[:,:] = 0
        start_vec[start_idx,0] = 1.0
    print(seq)
    txt = ''.join(idx_to_char[ix] for ix in seq)
    print(txt)

