import mnist_reader
import numpy as np
import neural
batch_size = 20
label_num = 10
data = mnist_reader.load_mnist(train_dir='data')
train = data.train

size = train.num_examples
idx = range(batch_size)
print size


l1 = neural.layer(784,50,"liner")
l2 = neural.layer(50,10,"softmax")

net = neural.network("softmax",learning_rate=0.0001)
net.add_layer(l1)
net.add_layer(l2)

for _ in range(5):
    images,labels = train.next_batch(batch_size)
    l = np.zeros([label_num,batch_size])
    images = images.T
    l[[labels],idx] = 1

    loss = net.learn(images,l)

    print loss

print l1.weights
print l2.weights