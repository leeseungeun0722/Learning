import sys, os

from yaml import load
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

# flatten = True -> 읽어들인 이미지는 1차원 넘파이 배열로 저장, 이미지를 표시할 때는 원래 형상인 28 * 28 크기로 다시 변형
# reshape() 매서드 -> 원하는 형상을 인수로 지정하면 넘파이 배열의 형상을 바꿀 수 있다
# 넘파이로 저장된 이미지 데이터를 PIL용 데이터 객체로 변환 , 변환은 image.fromarray()가 수행한다

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28,28)
print(img.shape)

img_show(img)