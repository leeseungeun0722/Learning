import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

# load_mnist함수 -> MINIST 데이터를 (훈련이미지, 훈련 레이블), (시험이미지, 시험레이블) 형태로 변환

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)