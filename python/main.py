import numpy as np
import pandas as pd
import ml
import json
import timeit
import multiprocessing
import time 
from functools import partial
import cProfile
import pstats

PARAMS_JSON_FILE_PATH = '../weights_and_biases.json'
TRAINING_DATA_FILE_PATH = '../digit-recognizer/train.csv'


def main():
    """Feedforward Neural Network (FNN) workflow."""     

    # run_train()

    infer()



def infer():
    """Make inferences using a trained model."""
    # Load model parameters and convert lists to numpy arrays.
    with open(PARAMS_JSON_FILE_PATH, 'r') as file:
        model_params = json.load(file)
    model = {}
    for key, vals in model_params.items():
        model.update({key: np.array(vals)})

    # W1 = np.array(data.get("W1"))
    # b1 = np.array(data.get("b1"))
    # W2 = np.array(data.get("W2"))
    # b2 = np.array(data.get("b2"))

    X_test, Y_test = load_dev_data()
    print("X_test sample:", X_test[0][:5])
    print("X_test shape:", X_test.shape)


    # run_predict(model, X_test, Y_test)

    # predict_sequential(model, X_test, Y_test)

    predict_multiprocess(model, X_test)

    # predict_multiprocess_pool(model, X_test, Y_test)


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


def profile(func):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        # profiler.print_stats(sort='cumulative')
        filename="profile_results"
        with open(filename, 'w') as f:
            ps = pstats.Stats(profiler, stream=f)
            ps.dump_stats(filename)
        return result
    return wrapper


    # cProfile.run('run_predict()', 'profile_results')
    # # do: snakeviz profile_results
    # p = pstats.Stats('profile_results')
    # p.sort_stats('cumulative')  #.print_stats(20)
    # print(p.print_stats(20))

    # results = pstats.Stats(cProfile.run("run_predict()"))
    # results.sort_stats(pstats.SortKey.TIME)
    # results.print_stats()


@profile
def run_predict(model, X_test, Y_test):
    dev_predictions = ml.make_predictions(X_test, model["W1"], model["b1"], model["W2"], model["b2"])
    print(dev_predictions[:10])
    print(ml.get_accuracy(dev_predictions, Y_test))


@profile
def predict_sequential(model, X_test, Y_test):
    """Sequential processing."""

    def f(X_test, Y_test, model): 
        dev_predictions = ml.make_predictions(X_test, model["W1"], model["b1"], model["W2"], model["b2"])
        acc = ml.get_accuracy(dev_predictions, Y_test)
        print(f"dev_predictions[:10]: {dev_predictions[:10]}, accuracy: {acc}")
        return acc

    # Create a partial function with fixed arguments
    func = partial(f, X_test, Y_test, model)

    # Number of times to run the method
    num_runs = 1000

    execution_time = timeit.timeit(func, number=num_runs) 
    print(f"Execution time: {execution_time} seconds")




def pred(X_test, Y_test, model, _):  # extra arg is for multiprocessing
    dev_predictions = ml.make_predictions(X_test, model["W1"], model["b1"], model["W2"], model["b2"])
    acc = ml.get_accuracy(dev_predictions, Y_test)
    return acc

@profile
def predict_multiprocess_pool(model, X_test, Y_test):
  
    num_runs = 1000 # Number of times to run the method
    num_processes = multiprocessing.cpu_count()

    # Create a partial function with fixed arguments
    func = partial(pred, X_test, Y_test, model)

    start_time = time.time()

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Run the method num_runs times in parallel
        results = pool.map(func, range(num_runs))
    
    end_time = time.time()
    print(f"Results: {results[:10]}... (showing first 10 results)")
    print(f"Multiprocessing execution time: {end_time - start_time} seconds")



@profile
def predict_multiprocess(model, X_test):

    start_time = time.time()
    processes = []

    # Create num_runs number of processes
    num_runs = 1000
    for i in range(num_runs): 
        p = multiprocessing.Process(target=worker, args=(i, X_test, model))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    end_time = time.time()
    print(f'Time taken: {end_time - start_time} seconds')


def worker(num, input, model):
    print(f'Worker {num} started')
    dev_predictions = ml.make_predictions(input, model["W1"], model["b1"], model["W2"], model["b2"])
    print("dev_predictions[:10]:", dev_predictions[:10])
    print(f'Worker {num} finished.')



if __name__ == "__main__":
    main()