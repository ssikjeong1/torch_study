# 자연어 처리를 위한 문자열 인코딩
#자연어를 모델에 입력하기 위해서는 숫자로 변환하는 과정이 필요합니다.
#이번 실습에서는 one-hot 인코딩을 구현해보고 단점들을 알아보겠습니다.
import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

# tokenizer는 파이썬에서 제공하는 텍스트 데이터 전처리 모듈입니다.
# tokenizer를 사용하면 문자열로 구성된 데이터를 다양한 형태로로 토큰화할 수 있습니다.
# set_token함수는 문자열들을 입력받아 그에 맞는 tokenizer 반환하는 함수입니다.
def set_token(texts):
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(texts)
    return tokenizer
  
# text2seq 함수와 seq2onehot는 하나의 문자열을 시퀀스 정보로 변환하고 one-hot 인코딩하는 함수입니다.
def text2seq(text, tokenizer):
    return tokenizer.texts_to_sequences([text])[0]

def seq2onehot(seq, num_word):
    return to_categorical(seq,num_classes=num_word+1) # 예약된 토큰을 위해 1자리를 추가로 사용

# 이제 임의의 문자열들을 처리하는 Tokenizer를 만들고, 텍스트를 one-hot 인코딩하는 과정을 실습해보겠습니다.
# 우선 간단한 영어문장 2개를 준비했습니다.
text1= "stand on the shoulders of giants"
text2= "I can stand on mountains"

# 이제 이 두 문장을 위한 Tokenizer를 만들어보겠습니다.
tokenizer = set_token([text1,text2])

# tokenizer는 text1과 text2에 있는 단어들을 토큰화할 수 있는 다양한 정보들을 가지고 있습니다.
print("단어 수: ", len(tokenizer.word_index))
print("단어 인덱스: ", tokenizer.word_index)

# 두 문장을 합쳐서 9개의 단어가 존재하고 이 단어들이 각각 1~9까지의 정수로 매핑되어 있는 것을 확인하실 수 있습니다.
# 이제 이 tokenizer를 통해 첫번째 문자열인 text2를 정수로 구성된 시퀀스 데이터로 변환하겠습니다.
seq = text2seq(text2, tokenizer)
print(seq)

# 5개의 단어로 구성된 text2가 단어의 인덱스로 구성된 데이터로 변환되었습니다.
# RNN모델에 이 데이터를 입력하기 위해서 one-hot 인코딩을 수행하겠습니다.
# 이 과정은 keras에서 제공하는 to_categorical 함수를 이용하겠습니다.
onehot1=seq2onehot(seq, len(tokenizer.word_index))
print(onehot1)
print("한 단어를 표현하는 길이:", len(onehot1[0]))

# 단어가 9가지를 나타내기 위한 9자리와 추후 모델에 입력하기 위해 남겨둔 1자리를 포함해서 10자리를 사용하고 있습니다.
# 상대적으로 간단한 문장들은 문제가 없어보입니다.
# 이번에는 좀 더 긴 문장들을 추가해보겠습니다.
# 아래의 `text3`과 `text4`는 추후에 다뤄볼 IMDB에 존재하는 문장의 일부입니다.
# 이 두 문장과 앞에서 사용했던 `text1``text2`를 모두 합쳐서 같은 과정을 수행하겠습니다.
text3 = "i have copy of this on vhs i think they the television networks should play this every year for the next twenty years so that we don't forget what was and that we remember not to do the same mistakes again like putting some people in the"
text4 = "he old neighborhood in serving time for an all to nice crime of necessity of course john heads back onto the old street and is greeted by kids dogs old ladies and his peer"

tokenizer2 = set_token([text1, text2, text3, text4])

print("단어 수: ", len(tokenizer2.word_index))
print("단어 인덱스: ", tokenizer2.word_index)

# 두 문장만 추가로 가져왔음에도 불구하고 이번에는 단어가 69가지로 훨씬 많아졌습니다. 
# 앞에서 다뤄본 `text2`를 다시 one-hot 인코딩까지 진행해보겠습니다.
seq2 = text2seq(text2, tokenizer2)
print(text2)
print(seq2)
onehot2 = seq2onehot(seq2,len(tokenizer2.word_index))
print(onehot2)

# 같은 문장이지만 훨씬 길게 표현되는 것을 확인하실 수 있습니다.
# 첫번째 단어인 `I`를 표현하는 벡터를 한번 보겠습니다.
print(onehot2[0])

# 이번에는 `I` 한글자를 위해 70차원의 벡터가 필요합니다.
# 또한, 70개의 숫자중 1개는 `1`이지만, 나머지 69개는 `0`으로 낭비되고 있습니다.
# 하지만 실제 자연어처리를 위한 데이터 셋은 더욱 다양한 단어를 포함하고 있고, one-hot 인코딩은 데이터 셋 전체에 존재하는 모든 단어의 종류만큼 길어질 것입니다.
