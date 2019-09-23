import numpy as np

def Linear(x, derivative=False):
    """
    Computes the element-wise Linear activation function for an array x
    inputs:
    x: The array where the function is applied
    derivative: if set to True will return the derivative instead of the forward pass
    """

    if derivative:  # Return the derivative of the function evaluated at x
        return np.ones_like(x)
    else:  # Return the forward pass of the function at x
        return x

def Sigmoid(x, derivative=False):
    """
    Computes the element-wise Sigmoid activation function for an array x
    inputs:
    x: The array where the function is applied
    derivative: if set to True will return the derivative instead of the forward pass
    """
    f = 1 / (1 + np.exp(-x))

    if derivative:  # Return the derivative of the function evaluated at x
        return f * (1 - f)
    else:  # Return the forward pass of the function at x
        return f

def Tanh(x, derivative=False):
    """
    Computes the element-wise Sigmoid activation function for an array x
    inputs:
    x: The array where the function is applied
    derivative: if set to True will return the derivative instead of the forward pass
    """
    f = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    if derivative:  # Return the derivative of the function evaluated at x
        return 1 - f ** 2
    else:  # Return the forward pass of the function at x
        return f

def ReLU(x, derivative=False):
    """
    Computes the element-wise Rectifier Linear Unit activation function for an array x
    inputs:
    x: The array where the function is applied
    derivative: if set to True will return the derivative instead of the forward pass
    """

    if derivative:  # Return the derivative of the function evaluated at x
        return (x > 0).astype(int)
    else:  # Return the forward pass of the function at x
        return np.maximum(x, 0)

class TwoWayDict(dict):
    def __setitem__(self, key, value):
        # Remove any previous connections with these values
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        """Returns the number of connections"""
        return dict.__len__(self) // 2


def squared_error(t, y, derivative=False):
    """
    Computes the squared error function and its derivative
    Input:
    t:      target (expected output)          (np.array)
    y:      output from forward pass (np.array, must be the same shape as t)
    derivative: whether to return the derivative with respect to y or return the loss (boolean)
    """
    if np.shape(t)!=np.shape(y):
        print("t and y have different shapes")
    if derivative: # Return the derivative of the function
        return (y-t)
    else:
        return 0.5*(y-t)**2


def forward_pass(x, NN, activations):
    """
    This function performs a forward pass. It saves lists for both affine transforms of units (z) and activated units (a)
    Input:
    x: The input of the network             (np.array of shape: (batch_size, number_of_features))
    NN: The initialized neural network      (tuple of list of matrices)
    activations: the activations to be used (list of functions, same len as NN)

    Output:
    a: A list of affine transformations, that is, all x*w+b.
    z: A list of activated units (ALL activated units including input and output).

    Shapes for the einsum:
    b: batch size
    i: size of the input hidden layer (layer l)
    o: size of the output (layer l+1)
    """
    #     print('activations', activations)
    z = [x]
    a = []

    for l in range(len(NN[0])):
        w = NN[0][l]
        b = NN[1][l]
        a.append(np.einsum('bi, io -> bo', z[l], w) + b)  # The affine transform x*w+b
        z.append(activations[l](a[l]))  # The non-linearity

    return a, z
