import tensorflow as tf
import numpy as np
data = open("input.txt").read()
vocab = list(set(data))
char_to_idx = {c:i for i,c in enumerate(vocab) }
idx_to_char = {i:c for i,c in enumerate(vocab) }
hidden_size = 10
vocab_size = len(vocab)
data_size = len(data)

learning_rate = 0.1
epoch = 1


inputs = tf.placeholder(shape=[1,vocab_size], dtype=tf.float32)
targets = tf.placeholder(shape=[1,vocab_size], dtype=tf.float32)
cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
softmax_weights = tf.Variable(tf.random_uniform([vocab_size,hidden_size]))
softmax_bias = tf.Variable(tf.zeros([vocab_size,1]))
init_state = cell.zero_state(1,tf.float32)
# state = init_state

h,state = cell(inputs,init_state)

out = tf.matmul(softmax_weights,h,transpose_b=True)+softmax_bias
out_max = tf.nn.softmax(out,dim=0)
out_index = tf.argmax(out_max,0)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(tf.reshape(out_max,[8,1]),tf.arg_max(targets,0))

train = tf.train.AdagradOptimizer(0.1).minimize(loss)

in_vec = np.zeros([vocab_size,1])
out_vec = np.zeros([vocab_size,1])

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for i in xrange(epoch):
        nstate = sess.run(init_state)
        total_loss = 0
        for j in xrange(data_size-1):
            in_char = data[i]
            in_idx = char_to_idx[in_char]
            out_char = data[i+1]
            out_idx = char_to_idx[out_char]
            in_vec[:,:] = 0
            in_vec[in_idx,0] = 1.0
            out_vec[:,:] = 0
            out_vec[out_idx,0] = 1.0
            nstate,l,t,m = sess.run([state,loss,train,out_max],feed_dict={inputs:in_vec.T,targets:out_vec.T,init_state:nstate})
            print m.shape
            total_loss += l
        print("{} loss after {} iter ".format(total_loss,i))
    start = data[0]
    start_idx = char_to_idx[start]
    start_vec = np.zeros([vocab_size, 1])
    start_vec[start_idx] = 1
    seq = []
    nstate = sess.run(init_state)
    for _ in range(30):
        seq.append(start_idx)
        y,nstate = sess.run([out_index,state],feed_dict={inputs:start_vec.T,state:nstate})
        start_idx = y[0]
        start_vec[:,:] = 0
        start_vec[start_idx,0] = 1
    print(seq)
    txt = ''.join(idx_to_char[ix] for ix in seq)
    print(txt)

