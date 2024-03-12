import numpy as np
from scipy import signal
from LoadMnistData import *
from Softmax import *
from ReLU import *
from Conv import *
from Pool import *
from MnistConv import *

Images, Labels = LoadMnistData('MNIST\\t10k-images-idx3-ubyte.gz', 'MNIST\\t10k-labels-idx1-ubyte.gz')
Images = np.divide(Images, 255)

W1 = 1e-2 * np.random.randn(9, 9, 20)
W5 = np.random.uniform(-1, 1, (100, 2000)) * np.sqrt(6) / np.sqrt(360 + 2000)
Wo = np.random.uniform(-1, 1, ( 10,  100)) * np.sqrt(6) / np.sqrt( 10 +  100)

X = Images[0:8000,:,:]
D = Labels[0:8000]

for epoch in range(3):
    print(epoch)
    W1, W5, Wo = MnistConv(W1, W5, Wo, X, D)

X = Images[8000:10000,:,:]
D = Labels[8000:10000]

acc = 0
N = len(D)
for k in range(N):
    x = X[k,:,:]
    
    y1 = Conv(x, W1)
    y2 = ReLU(y1)
    y3 = Pool(y2)
    y4 = np.reshape(y3, (-1,1))
    y5 = np.matmul(W5, y4)
    y6 = ReLU(y5)
    y7 = np.matmul(Wo, y6)
    y8 = Softmax(y7)
    
    i = np.argmax(y8)
    if i==D[k][0]:
        acc = acc + 1

acc = acc/N
print("Accuracy: ", acc)
