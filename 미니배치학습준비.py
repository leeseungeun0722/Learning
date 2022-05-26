import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

# load_mnist눈 MINIST 데이터셋을 읽어오는 함수이다

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000,)