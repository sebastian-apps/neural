import numpy as np
import pandas as pd
import ml
import json
import timeit
import multiprocessing
import time 
from functools import partial

PARAMS_JSON_FILE_PATH = '../weights_and_biases.json'
TRAINING_DATA_FILE_PATH = '../digit-recognizer/train.csv'


def main():
    """Feedforward Neural Network (FNN) workflow."""     

    # run_train()
 
    # run_predict()

    # predict_sequential()

    # predict_multiprocess()

    predict_multiprocess2()



def run_train():
    """Train the model and derive the weights and biases."""

    data = pd.read_csv(TRAINING_DATA_FILE_PATH)
    data = np.array(data)
    m, n = data.shape
    print("data.shape:", data.shape)

    np.random.shuffle(data)

    # Create the dev set.
    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255.

    # Print a sample of the data.
    print(Y_dev[:5])
    print(X_dev[0][:5])

    # Create the training set. 
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

    # Print a sample of the data.
    print("Y_dev:", Y_dev[:5])          
    print("X_dev sample:", X_dev[0][:5]) 

    dev_predictions = ml.make_predictions(X_dev, W1, b1, W2, b2)
    print(ml.get_accuracy(dev_predictions, Y_dev))

    ml.output_json(W1, b1, W2, b2, PARAMS_JSON_FILE_PATH)




def load_dev_data():
    """ Load the development set for testing. """
    data = pd.read_csv(TRAINING_DATA_FILE_PATH)
    data = np.array(data)
    m, n = data.shape
    print("train data.shape:", data.shape)
    np.random.shuffle(data)
    
    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255.
    # X_dev is transposed. Each column represents one image.
    return X_dev, Y_dev



def run_predict():
    """Make inferences using a trained model."""

    X_test, Y_test = load_dev_data()

    # Load model parameters
    with open(PARAMS_JSON_FILE_PATH, 'r') as file:
        data = json.load(file)

    # Print the dictionary keys verify its content
    print(data.keys())

    W1 = np.array(data.get("W1"))
    b1 = np.array(data.get("b1"))
    W2 = np.array(data.get("W2"))
    b2 = np.array(data.get("b2"))

    # print("Y_test:", Y_test[:5])
    print("X_test sample:", X_test[0][:5])
    print("X_test shape:", X_test.shape)

    dev_predictions = ml.make_predictions(X_test, W1, b1, W2, b2)

    print(dev_predictions[:10])

    print(ml.get_accuracy(dev_predictions, Y_test))



def predict_sequential():
    """Sequential processing."""

    def f(X_test, Y_test, W1, b1, W2, b2): 
        dev_predictions = ml.make_predictions(X_test, W1, b1, W2, b2)
        acc = ml.get_accuracy(dev_predictions, Y_test)
        print(f"dev_predictions[:10]: {dev_predictions[:10]}, accuracy: {acc}")
        return acc

    X_test, Y_test = load_dev_data()

    # Load model parameters.
    with open(PARAMS_JSON_FILE_PATH, 'r') as file:
        data = json.load(file)

    # Print the dictionary keys verify its content
    print(data.keys())

    W1 = np.array(data.get("W1"))
    b1 = np.array(data.get("b1"))
    W2 = np.array(data.get("W2"))
    b2 = np.array(data.get("b2"))

    # Create a partial function with fixed arguments
    func = partial(f, X_test, Y_test, W1, b1, W2, b2)

    # Number of times to run the method
    num_runs = 10

    execution_time = timeit.timeit(func, number=num_runs) 
    print(f"Execution time: {execution_time} seconds")




def pred(X_test, Y_test, W1, b1, W2, b2, _):  # extra arg is for multiprocessing
    dev_predictions = ml.make_predictions(X_test, W1, b1, W2, b2)
    acc = ml.get_accuracy(dev_predictions, Y_test)
    return acc


def predict_multiprocess():

    X_test, Y_test = load_dev_data()

    # Load model parameters
    with open(PARAMS_JSON_FILE_PATH, 'r') as file:
        data = json.load(file)

    # Print the dictionary keys verify its content
    print(data.keys())

    W1 = np.array(data.get("W1"))
    b1 = np.array(data.get("b1"))
    W2 = np.array(data.get("W2"))
    b2 = np.array(data.get("b2"))

    
    num_runs = 10 # Number of times to run the method
    num_processes = multiprocessing.cpu_count()

    # Create a partial function with fixed arguments
    func = partial(pred, X_test, Y_test, W1, b1, W2, b2)

    start_time = time.time()

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Run the method num_runs times in parallel
        results = pool.map(func, range(num_runs))
    
    end_time = time.time()
    print(f"Results: {results[:10]}... (showing first 10 results)")
    print(f"Multiprocessing execution time: {end_time - start_time} seconds")




def predict_multiprocess2():

    X_test, _ = load_dev_data()

    # Load data
    with open(PARAMS_JSON_FILE_PATH, 'r') as file:
        data = json.load(file)

    # Print the dictionary keys verify its content
    print(data.keys())

    W1 = np.array(data.get("W1"))
    b1 = np.array(data.get("b1"))
    W2 = np.array(data.get("W2"))
    b2 = np.array(data.get("b2"))

    start_time = time.time()
    processes = []

    # Create 10 processes
    for i in range(10): 
        p = multiprocessing.Process(target=worker, args=(i, X_test, W1, b1, W2, b2, ))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    end_time = time.time()
    print(f'Time taken: {end_time - start_time} seconds')


def worker(num, input, W1, b1, W2, b2):
    print(f'Worker {num} started')
    dev_predictions = ml.make_predictions(input, W1, b1, W2, b2)
    print("dev_predictions[:10]:", dev_predictions[:10])
    print(f'Worker {num} finished.')



if __name__ == "__main__":
    main()