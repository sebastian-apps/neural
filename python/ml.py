import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from icecream import ic
import json



# is full pipeline from backup file same as main2?
# I can add pipeline() if necessary.
def main2():
    # Feedforward Neural Network (FNN)     

    data = pd.read_csv('../digit-recognizer/train.csv')
    data = np.array(data)
    m, n = data.shape
    print("data.shape:", data.shape)

    np.random.shuffle(data)

    # dev set
    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255.

    print(Y_dev[:5])
    print(X_dev[0][:5])

    # training set 
    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255.
    _,m_train = X_train.shape

    print("Y_train:", Y_train)
    print("m_train:", m_train)

    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500, m)

    test_prediction(0, W1, b1, W2, b2, X_train, Y_train)
    test_prediction(1, W1, b1, W2, b2, X_train, Y_train)
    test_prediction(2, W1, b1, W2, b2, X_train, Y_train)
    test_prediction(3, W1, b1, W2, b2, X_train, Y_train)

    print("Y_dev:", Y_dev[:5])           # Print first 5 elements of Y_dev for brevity
    print("X_dev sample:", X_dev[0][:5]) # Print first 5 elements of the first row of X_dev for brevity

    dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
    print(get_accuracy(dev_predictions, Y_dev))

    output_json(W1, b1, W2, b2)



def gradient_descent(X, Y, alpha, iterations, m):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y, m)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    print("\ninit_params():")
    print(f"type(W1):{type(W1)}, type(b1): {type(b1)}, type(W2): {type(W2)}, type(b2): {type(b2)}")
    return W1, b1, W2, b2

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y, m):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1).reshape(-1, 1)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1).reshape(-1, 1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2, X_train, Y_train):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


def output_json(W1, b1, W2, b2):

    weights_biases = {
        "W1": W1.tolist(),
        "b1": b1.tolist(),
        "W2": W2.tolist(),
        "b2": b2.tolist(),
    }

    json_string = json.dumps(weights_biases, indent=4)

    filename = '../weights_and_biases.json'
    with open(filename, 'w') as file:
        file.write(json_string)




