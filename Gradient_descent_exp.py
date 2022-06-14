import numpy as np
import matplotlib.pyplot as plt



def main():
    
    gradient()
    
def gradient():
    '''
    지시 사항: 함수 안에 f함수의 미분을 계산해 df부분을 채우세요.
    '''
    cur_x = 3 # 현재 x
    lr = 0.01 # Learning rate
    threshold = 0.000001 # 알고리즘의 while문을 조절하는 값.
    previous_step_size = 1 #
    max_iters = 10000 # 최대 iteration 횟수
    iters = 0 #iteration counter
    f = lambda x: np.exp(x+1) # 우리가 구하고자하는 함수의 미분함수
    df = None
    
    history_x = []
    history_y = []
    while previous_step_size > threshold and iters < max_iters:
        prev_x = cur_x # 현재 x를 prev로 저장합니다.
        cur_x = cur_x - lr * df(prev_x) #Grad descent
        previous_step_size = abs(cur_x - prev_x) #x의 변화량을 구합니다.
        iters = iters+1 #iteration count
        print("Iteration",iters,"\nx 값은 ",cur_x, "\ny 값은 ", f(cur_x)) #Print iterations
        history_x.append(None)
        history_y.append(None)

    plt.plot(history_x, history_y, marker='o')
    plt.show()

if __name__ == "__main__":
    main()
