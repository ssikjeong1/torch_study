import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(100)


def main():
    data = pd.read_csv('./data.csv', header=None)
    X = np.array(data[[0,1]])
    y = np.array(data[2])
    plot_points(X,y)

    plt.savefig("plot.png")
    elice_utils.send_image("plot.png")
    
    epochs = 100
    learnrate = 0.01
    train(X, y, epochs, learnrate, True)



def plot_points(X, y):
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'blue', edgecolor = 'k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'red', edgecolor = 'k')

def display(m, b, color='g--'):
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    x = np.arange(-10, 10, 0.1)
    plt.plot(x, m*x+b, color)

    # Activation (sigmoid) function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def function(features, weights, bias):
    return sigmoid(np.dot(features, weights) + bias)

def objective(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)

def update_weights(x, y, weights, bias, learnrate):
    output = None # function 함수를 통해 나오는 출력을 적어주세요.
    d_error = -(y - output)
    weights -= None # d_error * x는 gradient 값입니다. gradient에는 learning rate을 곱해주어서 weight를 업데이트 합니다.
    bias -= None # weight와 마찬가지로 learningrate을 곱해서 update 해주세요.
    return weights, bias


def train(features, targets, epochs, learnrate, graph_lines=False):
    
    errors = []
    n_records, n_features = features.shape
    last_loss = None
    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)
    bias = 0
    for e in range(epochs):
        del_w = np.zeros(weights.shape)
        for x, y in zip(features, targets):
            output = function(x, weights, bias)
            error = objective(y, output)
            weights, bias = update_weights(x, y, weights, bias, learnrate)
        
        # Printing out the log-loss error on the training set
        out = function(features, weights, bias)
        loss = np.mean(objective(targets, out))
        errors.append(loss)
        if e % (epochs / 10) == 0:
            print("\n========== Epoch", e,"==========")
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            predictions = out > 0.5
            accuracy = np.mean(predictions == targets)
            print("Accuracy: ", accuracy)
        if graph_lines and e % (epochs / 10) == 0:
            display(-weights[0]/weights[1], -bias/weights[1])
            


    # Plotting the solution boundary
    plt.title("Solution boundary")
    display(-weights[0]/weights[1], -bias/weights[1], 'black')

    # Plotting the data
    plot_points(features, targets)
    plt.savefig("plot1.png")
    elice_utils.send_image("plot1.png")

    # Plotting the error
    plt.cla()
    plt.title("Error Plot")
    plt.xlabel('Number of epochs')
    plt.ylabel('Error')
    plt.plot(errors)

    plt.savefig("plot2.png")
    elice_utils.send_image("plot2.png")


if __name__ == "__main__":
    main()
