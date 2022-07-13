## Tensorflow로 conv2d 사용하기

#Tensorflow의 2차원 Convolution은 `tf.nn.conv2d`을 사용합니다. 
#간단한 행렬을 입력하면서 Padding과 Stride의 동작을 살펴보겠습니다.

import tensorflow as tf
from tensorflow import keras

# 입력은 1로 구성된 3x3 크기의 간단한 행렬입니다.
# 모델에 입력할때는 여기에 색상의 차원수, 채널수 2가지를 추가해야 합니다.

inp = tf.ones((1, 3, 3, 1)) # 3x3 x1 이미지가 1개 (1,높이,너비,1)
print(inp)

# [[[[1][1][1]]
#   [[1][1][1]]
#   [[1][1][1]]]]

# Filter는 1로 가득찬 2x2의 크기를 가진 행렬 1개를 사용하겠습니다.
filter = tf.ones((2, 2, 1, 1)) # 2x2 x1 짜리 필터가 1개 
print(filter)

# [ [ [[1.]],[[1.]] ],
#   [ [[1.]],[[1.]] ] ] 

# strides 는 [높이, 너비]의 형식으로 입력합니다.
# 이번에는 1칸씩 이동하도록 1, 1을 입력합니다.
strides = [1, 1] # [높이, 너비]

# 이제 준비된 입력값, filter, stride로 Convolution 연산을 수행하겠습니다.
output = tf.nn.conv2d(inp, filter, strides,padding = 'VALID') # padding을 'VALID'으로 설정 = 패딩을 하지 않음
print(output)
# [[  [[4.] [4.]]
#     [[4.] [4.]]  ]], shape=(1, 2, 2, 1), dtype=float32)

# Padding이 없는 상태에서 Convolution을 수행하니 입력의 크기(3x3)보다 출력의 크기(2x2)가 작아졌습니다.
# 만약 여기에 한번더 Convolution을 적용하면 어떻게 될까요??
output = tf.nn.conv2d(output, filter, strides, padding = 'VALID') # 한번 더 적용
print(output)

# 이번에는 (2x2)의 크기에서 1칸으로 줄어들었습니다.
# 이처럼 padding을 적용하지 않고 Convolution을 적용하면 크기가 점점 줄어들게 됩니다.
# 이번에는 padding옵션을 'VALID'가 아닌 'SAME'으로 설정해보겠습니다.
output = tf.nn.conv2d(inp, filter, strides,padding = 'SAME') # padding을 'SANE'으로 설정 = 입력과 출력의 형태가 같도록 패딩을 적용
print(output)
#  [[ [[4.] [4.] [2.]]
#     [[4.] [4.] [2.]]
#     [[2.] [2.] [1.]] ]], shape=(1, 3, 3, 1), dtype=float32)

# 이번엔 크기가 줄어들지 않고 동일하게 3x3의 크기로 출력되었습니다.
# Convolution Layer에서 padding을 'SAME'으로 설정하면 여러번 연산해도 그 크기는 줄어들지 않습니다.
# 이번에는 padding을 직접 설정해서 전달해보겠습니다.
# 위,아래,오른쪽,왼쪽에 각각 한 칸씩 추가해보겠습니다.
padding = [[0, 0], [1, 1], [1, 1], [0, 0]] # [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
output1 = tf.nn.conv2d(inp, filter, strides, padding = padding) # 명시적으로 padding 전달하기
print(output1)
# [[ [[1.]  [2.]  [2.]  [1.]]
#    [[2.]  [4.]  [4.]  [2.]]
#    [[2.]  [4.]  [4.]  [2.]]
#    [[1.]  [2.]  [2.]  [1.]] ]]

# 이번에는 (3x3) 행렬의 위,아래,왼쪽,오른쪽에 각각 한 칸을 추가해서 5x5의 크기가 되었습니다.
# 다시 이 행렬에 2x2의 필터 1개으로 구성된 conv2d layer를 통과하니 4x4로 바뀌었습니다.
# padding은 conv2d layer에 값을 전달하는 방법도 있지만, 직접 padding을 적용하는 방법도 있습니다.
# 이번에는 tf.pad 함수를 이용하여 직접 padding을 적용하고 이것을 conv2d layer에 입력해보겠습니다.
pad_inp = tf.pad(inp, padding) # padding을 적용하는 함수 pad
# print(pad_inp)
# [[ [[0.] [0.] [0.] [0.] [0.]]
#    [[0.] [1.] [1.] [1.] [0.]]
#    [[0.] [1.] [1.] [1.] [0.]]
#    [[0.] [1.] [1.] [1.] [0.]]
#    [[0.] [0.] [0.] [0.] [0.]] ]], shape=(1, 5, 5, 1), dtype=float32)

output2 = tf.nn.conv2d(pad_inp, filter, strides, padding = 'VALID') # 'VALID' : padding을 하지 않음(직접 padding을 적용하고 입력하기 위해)
# 미리 패딩을 적용해둔 pad_inp를 입력
print("output2\n",*output2)
# [[  [[1.] [2.] [2.] [1.]]
#     [[2.] [4.] [4.] [2.]]
#     [[2.] [4.] [4.] [2.]]
#     [[1.] [2.] [2.] [1.]]  ]], shape=(1, 4, 4, 1), dtype=float32)
print("output1\n",output1)

# 우리는 방금 padding을 적용하는 방법 두가지를 실습했습니다.
# * conv2d의 매개변수인 padding을 이용하는 방법 (output1)
# * 직접 tf.pad 함수를 이용하여 패딩을 적용하고 입력하는 방법이 있습니다. (output2)

# 위에서 두 방식의 결과가 차이가 없다는 것까지 확인하실 수 있습니다.
# ----
# ## Tensorflow.Keras로 Conv2D 사용하기
# 이번에는 Tensorflow.Keras를 사용할 때 차이점을 알아보겠습니다.
# 이번에도 편의를 위해 입력 값을 1로 구성된 간단한 행렬로 설정하겠습니다.

input_shape=(1, 3, 3, 1)

x = tf.ones(input_shape) # 3x3 x1 이미지가 1개 (1, 높이, 너비, 1)
print(x)

# [[ [[1.] [1.] [1.]]
#    [[1.] [1.] [1.]]
#    [[1.] [1.] [1.]]  ]], shape=(1, 3, 3, 1), dtype=float32)

# `tf.keras.layers.Conv2D` 와 `tf.nn.conv2d` 매개변수의 이름이 약간씩 다릅니다.
# * filters : 필터의 갯수입니다. 우리는 1개의 필터를 사용하므로 1을 전달합니다.
# * kernel_size : kernel의 크기 즉, filter의 형태를 (높이, 너비) 형태로 전달합니다.
# * strides `tf.nn.conv2d`와 동일하게 사용합니다.
# * padding : `tf.nn.conv2d`과 비슷하지만 모두 소문자로 전달합니다. 'same', ' valid'의 동작은 같습니다.
# * activation : 활성함수는 `'relu'`를 전달하겠습니다.
# * input_shape : keras로 구성한 모델의 가장 첫번째 Layer에는 입력의 형태를 전달해야 합니다.

y = tf.keras.layers.Conv2D( filters = 1, # 필터의 갯수 
                            kernel_size = [2, 2], # "kernel_size = 2" 와 같은 의미 (높이, 너비)
                            strides = (1, 1), 
                            padding = 'same', # keras.layers.Conv2D 의 padding은 소문자 'same', 'valid'
                            activation = 'relu', 
                            input_shape = input_shape[1:]) (x) # 입력 : x
print(y)
# [[ [[0.36910588] [0.36910588] [0.54728895]]
#    [[0.36910588] [0.36910588] [0.54728895]]
#    [[0.8551657 ] [0.8551657 ] [0.6025906 ]] ]], shape=(1, 3, 3, 1), dtype=float32)

# Keras에서 가중치를 무작위 값으로 초기화하는 과정까지 수행해서 값은 조금 다르게 나타났습니다.
# 하지만 Tensor의 형태는 동일하게 유지되었습니다.
