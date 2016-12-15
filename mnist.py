import mnist_reader
import numpy as np
import neural
batch_size = 100
label_num = 10
data = mnist_reader.load_mnist(train_dir='data')
train = data.train
test = data.test

size = train.num_examples
idx = range(batch_size)
print size


l1 = neural.layer(784,100,"liner")
l2 = neural.layer(100,10,"softmax")

net = neural.network("softmax",learning_rate=0.00015)
net.add_layer(l1)
net.add_layer(l2)

for _ in range(2*size//batch_size):
    images,labels = train.next_batch(batch_size)
    l = np.zeros([label_num,batch_size])
    images = images.T
    l[[labels],idx] = 1

    loss = net.learn(images,l)

    print loss
test_size = len(test.images)
correct = 0
for _ in range(test_size):
    images,labels = test.next_batch(1)
    images = images.T
    o = net.eval(images)
    n = np.argmax(o,axis=0)
    print n
    print labels
    if(n[0] == labels[0] ):
        correct += 1

print "{} correct, rate is {}".format(correct,correct*1.0/test_size)