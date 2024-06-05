import numpy as np
import pandas as pd
import ml

def main():
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

    W1, b1, W2, b2 = ml.gradient_descent(X_train, Y_train, 0.10, 500, m)

    ml.test_prediction(0, W1, b1, W2, b2, X_train, Y_train)
    ml.test_prediction(1, W1, b1, W2, b2, X_train, Y_train)
    ml.test_prediction(2, W1, b1, W2, b2, X_train, Y_train)
    ml.test_prediction(3, W1, b1, W2, b2, X_train, Y_train)

    print("Y_dev:", Y_dev[:5])           # Print first 5 elements of Y_dev for brevity
    print("X_dev sample:", X_dev[0][:5]) # Print first 5 elements of the first row of X_dev for brevity

    dev_predictions = ml.make_predictions(X_dev, W1, b1, W2, b2)
    print(ml.get_accuracy(dev_predictions, Y_dev))

    ml.output_json(W1, b1, W2, b2)


main()