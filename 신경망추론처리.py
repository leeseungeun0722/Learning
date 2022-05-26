import sys,os
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

# MINIST 데이터셋을 이용한 추론을 수행하는 신경망을 구현
# 입력층 뉴런 784개 -> 이미지크기 28 * 28 = 784
# 출력층 뉴런 10개 0에서 9까지의 숫자를 구분하는 문제'
# init_network() -> pickle 파일인 sample_weight.pkl에 저장된 학습된 가중치 매개변수를 읽기, 가중치와 편향 매개변수가 딕셔너리 변수로 저장
# predict() -> network : 가중치가 포함된 신경망 모델, x : 입력값, 순전파 과정을 통해 x에 대한 분류 진행

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl",'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1,W2,W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = softmax(a3)
    return y

x,t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):

    y = predict(network, x[i])
    p = np.argmax(y)

    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy : " + str(float(accuracy_cnt) / len(x)))