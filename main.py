import xml.etree.ElementTree as ET
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import unicodedata
import matplotlib.pyplot as plt
from parameters import get_parameters
import joblib


def sigmoid(x):
    """
    Compute the sigmoid of x
    """
    s = 1 / (1 + np.exp(-x))
    return s


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)


def strip_diacritics(text):
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)


def str_to_char_codes(text) -> list:
    return list(map(lambda x: ord(x), list(strip_diacritics(text.lower()))))


def char_codes_to_str(codes) -> list:
    return ''.join(map(lambda x: chr(x), codes))


def load_words():
    xml = ET.parse('deu-ita.tei')
    root = xml.getroot()

    german = list(
        map(lambda x: str_to_char_codes(x.text),
            filter(lambda x: x.tag == 'orth', root.iter())))
    # and len(x.text) < 6
    italian = list(
        map(lambda x: str_to_char_codes(x.text),
            filter(lambda x: x.tag == 'quote', root.iter())))
    return german, italian


def load_data():
    german_words, italian_words = load_words()

    words = german_words + italian_words
    # words = words.reshape(words.shape[0], 1)

    max_char_val = max(map(lambda word: max(word), words))

    # np.array(german).reshape((len(german), 1))
    words_2d = np.zeros(
        [len(words), max(map(lambda x: len(x), words))], dtype=int)
    for i, j in enumerate(words):
        words_2d[i][0:len(j)] = j
    words_2d = words_2d / max_char_val
    # words_2d = words_2d.T

    labels = np.append(np.zeros((1, len(german_words)), dtype=int),
                       np.ones((1, len(italian_words)), dtype=int)).reshape(
                           (words_2d.shape[0], 1))

    X_train, X_test, Y_train, Y_test = train_test_split(words_2d,
                                                        labels,
                                                        test_size=0.2)
    return X_train.T, X_test.T, Y_train.T, Y_test.T, max_char_val


def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """

    # we set up a seed so that your output matches ours although the initialization is random.
    # np.random.seed(2)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    return parameters


def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

    return A2, cache


def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)
    
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    [Note that the parameters argument is not used in this function, 
    but the auto-grader currently expects this parameter.
    Future version of this notebook will fix both the notebook 
    and the auto-grader so that `parameters` is not needed.
    For now, please include `parameters` in the function signature,
    and also when invoking this function.]
    
    Returns:
    cost -- cross-entropy cost given equation (13)
    
    """

    m = Y.shape[1]  # number of example

    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    cost = -np.sum(logprobs) / m

    cost = float(
        np.squeeze(cost))  # makes sure cost is the dimension we expect.
    # E.g., turns [[17]] into 17
    assert (isinstance(cost, float))

    return cost


def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.
    
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]

    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache["A1"]
    A2 = cache["A2"]

    # Backward propagation: calculate dW1, db1, dW2, db2.
    dZ2 = A2 - Y
    dW2 = (np.dot(dZ2, A1.T)) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = (W2.T * dZ2) * (1 - np.power(A1, 2))
    dW1 = (np.dot(dZ1, X.T)) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    """
    Updates parameters using the gradient descent update rule given above
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # Update rule for each parameter
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    return parameters


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    # np.random.seed(3)
    n_x = X.shape[0]
    n_y = Y.shape[0]

    # Initialize parameters
    parameters = initialize_parameters(n_x, n_h, n_y)

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)

        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)

        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)

        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads)

        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters


def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, _ = forward_propagation(X, parameters)
    predictions = A2 > 0.5

    return predictions


def main():
    train = True

    X_train, X_test, Y_train, Y_test, max_char_val = load_data()

    if train:
        parameters = nn_model(X_train,
                              Y_train,
                              n_h=6,
                              num_iterations=1000,
                              print_cost=True)
        from datetime import datetime
        now = datetime.now()
        filename = 'parameters/{}.param'.format(
            now.strftime('%Y-%m-%d %H:%M:%S'))
        joblib.dump(parameters, filename)
    else:
        parameters = get_parameters()[2]
        # parameters = joblib.load(filename)

    correct = 0
    total = 0
    for index in range(len(X_test.T)):
        x_test = X_test.T[index].reshape((X_test.shape[0], 1))
        y_test = Y_test.T[index]
        yhat = predict(parameters, x_test)
        correct += 1 if y_test == yhat else 0
        total += 1

    print("Accuracy: {} ({} right out of {})".format(correct / total, correct,
                                                     total))

    while True:
        my_word = input('Enter a word: ')
        word_chars = np.array(str_to_char_codes(my_word)).reshape(
            (len(my_word), 1))

        word_array = np.zeros((X_test.shape[0], 1), dtype=int)
        for (i, c) in enumerate(word_chars):
            word_array[i] = c

        word_array = word_array / max_char_val

        prediction = predict(parameters, word_array)
        print('My magic brain thinks this word is {}\n'.format(
            'italian' if prediction[0][0] else 'german'))


if __name__ == "__main__":
    main()
