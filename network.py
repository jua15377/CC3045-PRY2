import numpy as np

# adapted from: http://neuralnetworksanddeeplearning.com/chap1.html
# by Michael Nielsen
class Network(object):

    def __init__(self, layers):
        self.num_layers = len(layers)
        self.layers = layers
        # init random values for w y b
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            # g((theta + X) + b)
            a = sigmoid(np.dot(w, a)+b)
        return a

    def minibach_gradient_decent(self, data, epochs, mini_batch_size, eta):
        data = list(data)
        data_size = len(data)
        # create mini baches
        for j in range(epochs):
            mini_batches = [data[k:k + mini_batch_size] for k in range(0, data_size, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)


    def update_mini_batch(self, mini_batch, eta):
        # inicial values
        bach_size = len(mini_batch)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            # do backprop for the bach
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # update weight and biases
        self.weights = [w-(eta/bach_size)*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/bach_size)*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        '''
        :param x: image data
        :param y: expected output
        :return: tupla with the values to updates
        '''
        # init values
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x # set first layer
        activations = [x] # activation
        zs = [] # values z
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward
        # for the las layer
        delta = cost_derivative(activations[-1], y) * sigmoid_derivate(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l] #fillin z in reverse
            sp = sigmoid_derivate(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def test_performance(self, data):
        '''
        run feedforward over each of the immputs anthen gives the amount of correct predictions
        :param data: set to be tested
        :return: amount of correct answers
        '''
        results = []
        for data in data:
            test = (np.argmax(self.feedforward(data[0])), np.argmax(data[1]) )
            results.append(test)
        total = 0
        for (x, y) in results:
            if x == y: total += 1
        return total


    def predict(self, input):
        result = self.feedforward(input)
        return np.argmax(result), result

    def load_training(self, w, b):
        '''
        loads the weights and biases pre trained
        :param w: np.array from saved file
        :param b: np.array from saved file
        :return: None
        '''
        self.weights = list(w)
        self.biases = list(b)

    def save_training(self, version):
        '''
        call this function to save the trained network
        :param version: this specify the sufix in the output file
        :return: None
        '''
        np.save('memory/weights_'+ version, self.weights)
        np.save('memory/biases_' + version, self.biases)


def cost_derivative(output_activations, y):
    return (output_activations - y)

def cost_function(x, y):
    n = len(ou)
    return (1.0 / 2.0 * n) * sum((x - y)) ** 2

def sigmoid(z):
    '''
    sigmoid function.
    :param z: a vector to apply the function.
    :return: new vector with the results of the sigmoid
    '''
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_derivate(z):
    return sigmoid(z) * (1-sigmoid(z))
