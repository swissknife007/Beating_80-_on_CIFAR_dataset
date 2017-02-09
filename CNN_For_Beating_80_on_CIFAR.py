"""

Source Code for Homework 3 of ECBM E4040, Fall 2016, Columbia University
Instructor: Prof. Zoran Kostic

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""

from __future__ import print_function

import timeit
import inspect
import sys
import numpy
from theano.tensor.nnet import conv
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
import theano.tensor.nnet

import numpy
import numpy
import timeit
import theano
import theano.tensor as T
from theano.tensor.signal import downsample, pool
import numpy
import timeit
import theano
import theano.tensor as T
from theano.tensor.signal import downsample, pool

from hw3_utils import shared_dataset, load_data
#from hw2_nn_new import LogisticRegression, HiddenLayer, LeNetConvPoolLayer, train_nn
import sys,os
import os
import sys
import numpy
import scipy.io
import tarfile
import theano
import theano.tensor as T


def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

def floatX(X):
    return numpy.asarray(X, dtype=theano.config.floatX)

def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(numpy.asarray(X, dtype=dtype), name=name)
def shared_zeros(shape, dtype=theano.config.floatX, name=None):
    return sharedX(numpy.zeros(shape), dtype=dtype, name=name)

def translate_image(X, translate_p = 0):
    lenX = X.shape[0]
    #print X.shape
    iter = 0
    deepX = X[:]
    while iter < lenX:
        if(numpy.random.random() >= translate_p):
            iter = iter + 1
            continue
        im=numpy.reshape(X[iter],(3,32,32))
        im = im.transpose(1,2,0)
        randx = numpy.random.randint(0,6)
        randy = numpy.random.randint(0,6)
        if(numpy.random.random() > 0.5):
            randx = randx * -1
        if(numpy.random.random() > 0.5):
            randy = randy * -1
        im2 = scipy.ndimage.shift(im,[randx,randy,0])
        deepX[iter] = im2.transpose(2,0,1).flatten()
        iter = iter + 1
    return deepX
def rotate_image(X, rotate_p = 0):
    lenX = X.shape[0]
    #print X.shape
    iter = 0
    deepX = X[:]
    while iter < lenX:
        if(numpy.random.random() >= rotate_p):
            iter = iter + 1
            continue
        randx = numpy.random.randint(0,15)    
        theta = randx
        if(numpy.random.random() > 0.5):
            theta = theta *-1
        im = numpy.reshape(X[iter],(3,32,32))
        im = im.transpose(1,2,0)
        im2 = scipy.ndimage.rotate(im, theta+0.001, reshape=False)
        deepX[iter] = im2.transpose(2,0,1).flatten()
        iter = iter + 1
    return deepX


def noise_image(X, gaussian_noise = True, noise_p = 0):
    lenX = X.shape[0]
    #print X.shape
    iter = 0
    deepX = X[:]
    while iter < lenX:
        if(numpy.random.random() >= noise_p):
            iter = iter + 1
            continue
        randx = numpy.random.randint(0,6)    
        theta = randx
        if(numpy.random.random() > 0.5):
            theta = theta *-1
        im = numpy.reshape(X[iter],(3,32,32))
        im = im.transpose(1,2,0)
        im2 = im
        if(gaussian_noise):
            noise = numpy.random.normal(0, 0.025, [32,32,3])
            im2 = noise + im2
        else:
            noise = numpy.random.uniform(low=-0.025, high=0.025, size=[32,32,3]) 
            im2 = im2 + noise
           
        deepX[iter] = im2.transpose(2,0,1).flatten()
        iter = iter + 1
    return deepX
    
#Implement a convolutional neural network with the translation method for augmentation
#def test_lenet_translation():


#Problem 2.2
#Write a function to ad#d roatations
#def rotate_image():
#Implement a convolutional neural network with the rotation method for augmentation
#def test_lenet_rotation():

#Problem 2.3
#Write a function to flip images
def flip_image(X, flip_p = 0):
    lenX = X.shape[0]
    #print X.shape
    iter = 0
    deepX = X[:]
    while iter < lenX:
        if(numpy.random.random() >= flip_p):
            iter = iter + 1
            continue
        temp = numpy.reshape(X[iter],(3,32,32)).transpose(1,2,0) 
        deepX[iter] = numpy.fliplr(temp).transpose(2,0,1).flatten()
        iter = iter + 1
    return deepX

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

        
def drop(input, p=0.5):
    """
    :type input: numpy.array
    :param input: layer or weight matrix on which dropout is applied

    :type p: float or double between 0. and 1.
    :param p: p probability of NOT dropping out a unit, therefore (1.-p) is the drop rate.

    """
    rng = numpy.random.RandomState(1234)
    srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
    return input * mask

class DropoutHiddenLayer(object):
    def __init__(self, is_train, rng, input=1, n_in=1, n_out = 500,W=None, b=None,
                 activation=T.tanh, p=0.5):
        # type: (object, object, object, object, object, object, object, object, object) -> object
        """
        Hidden unit activation is given by: activation(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type is_train: theano.iscalar
        :param is_train: indicator pseudo-boolean (int) for switching between training and prediction

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer

        :type p: float or double
        :param p: probability of NOT dropping out a unit
        """
        self.input = input

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                     high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)


        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b

        output = activation(lin_output)

        # multiply output and drop -> in an approximation the scaling effects cancel out
        train_output = drop(output,p)

        #is_train is a pseudo boolean theano variable for switching between training and prediction
        self.output = T.switch(T.neq(is_train, 0), train_output, p*output)

        # parameters of the model
        self.params = [self.W, self.b]


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        print('....image_shape....')
        print(image_shape)
        print('input shape....')
        print(filter_shape)
        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode = 'half'
        )

        # pool each feature map individually, using maxpooling
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input



def test_mynet(learning_rate=0.001, n_epochs=80,
                    dataset='mnist.pkl.gz',
                    nkerns=[20, 50], batch_size=100, flip_p = 0, rotate_p = 0, translate_p = 0, noise_p = 0 ):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)
    
    ds_rate = None
    ''' Loads the SVHN dataset

    :type ds_rate: float
    :param ds_rate: downsample rate; should be larger than 1, if provided.

    :type theano_shared: boolean
    :param theano_shared: If true, the function returns the dataset as Theano
    shared variables. Otherwise, the function returns raw data.
    '''
    if ds_rate is not None:
        assert(ds_rate > 1.)

    # Download the CIFAR-10 dataset if it is not present
    def check_dataset(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        #f_name = new_path.replace("src/../data/%s"%dataset, "data/") 
        f_name = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data"
        )
        if (not os.path.isfile(new_path)):
            from six.moves import urllib
            origin = (
                'https://www.cs.toronto.edu/~kriz/' + dataset
            )
            print('Downloading data from %s' % origin)
            urllib.request.urlretrieve(origin, new_path) 
             
        tar = tarfile.open(new_path)
        file_names = tar.getnames()
        for file_name in file_names:
            tar.extract(file_name,f_name)
        tar.close()              
        
        return f_name
    
    f_name=check_dataset('cifar-10-matlab.tar.gz')
    
    train_batches=os.path.join(f_name,'cifar-10-batches-mat/data_batch_1.mat')
    
    
    # Load data and convert data format
    train_batches=['data_batch_1.mat','data_batch_2.mat','data_batch_3.mat','data_batch_4.mat','data_batch_5.mat']
    train_batch=os.path.join(f_name,'cifar-10-batches-mat',train_batches[0])
    train_set=scipy.io.loadmat(train_batch)
    train_set['data']=train_set['data']/255.
    for i in range(4):
        train_batch=os.path.join(f_name,'cifar-10-batches-mat',train_batches[i+1])
        temp=scipy.io.loadmat(train_batch)
        train_set['data']=numpy.concatenate((train_set['data'],temp['data']/255.),axis=0)
        train_set['labels']=numpy.concatenate((train_set['labels'].flatten(),temp['labels'].flatten()),axis=0)
    
    test_batches=os.path.join(f_name,'cifar-10-batches-mat/test_batch.mat')
    test_set=scipy.io.loadmat(test_batches)
    test_set['data']=test_set['data']/255.
    test_set['labels']=test_set['labels'].flatten()
    
    train_set=(train_set['data'],train_set['labels'])
    test_set=(test_set['data'],test_set['labels'])
    

    # Downsample the training dataset if specified
    train_set_len = len(train_set[1])
    if ds_rate is not None:
        train_set_len = int(train_set_len // ds_rate)
        train_set = [x[:train_set_len] for x in train_set]

    # Extract validation dataset from train dataset
    valid_set = [x[-(train_set_len//5):] for x in train_set]
    train_set = [x[:-(train_set_len//5)] for x in train_set]

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.
    theano_shared=True
    if theano_shared:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    else:
        rval = [train_set, valid_set, test_set]

    #return rval
    datasets = rval

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    training_enabled = T.iscalar('training_enabled')
    # start-snippet-1
    mydata = T.matrix('mydata')
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    print(layer0_input.shape)


    
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(128, 3, 3, 3)
    )
    
    print('layer 0 constructed....')
    print(layer0.output)
    layer01 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, 128, 32, 32),
        filter_shape=(128, 128, 3, 3)
       
    )
    
    layer02 = LeNetConvPoolLayer(
        rng,
        input=layer01.output,
        image_shape=(batch_size, 128, 32, 32),
        filter_shape=(32, 128, 3, 3)
        
    )
    '''
    layer03 = LeNetConvPoolLayer(
        rng,
        input=layer02.output,
        image_shape=(batch_size, 32, 32, 32),
        filter_shape=(32, 32, 5, 5)
        
    )
    layer04 = LeNetConvPoolLayer(
        rng,
        input=layer03.output,
        image_shape=(batch_size, 32, 32, 32),
        filter_shape=(32, 32, 5, 5)
        
    )
    layer05 = LeNetConvPoolLayer(
        rng,
        input=layer04.output,
        image_shape=(batch_size, 32, 32, 32),
        filter_shape=(32, 32, 5, 5)
       
    )
    '''
    layer06 = theano.tensor.signal.pool.pool_2d(layer02.output, (2,2), ignore_border = True)
    
    
    
    print('layer 01 constructed....')
    print(layer01)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer06,
        image_shape=(batch_size, 32, 16, 16),
        filter_shape=(32, 32, 3, 3)
    
    )
    
    layer2 = theano.tensor.signal.pool.pool_2d(layer1.output, (2,2), ignore_border = True)
    

    layer3 = LeNetConvPoolLayer(rng, input = layer2, image_shape = (batch_size,32,8,8),filter_shape=(32,32,3,3), poolsize=(1,1))
    
    layer4 = theano.tensor.signal.pool.pool_2d(layer3.output, (2,2), ignore_border = True)
    

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer5_input = layer4.flatten(2)

    # construct a fully-connected sigmoidal layer
    
    layer5 = DropoutHiddenLayer(
        
        is_train= training_enabled,
        rng=rng,
        input=layer5_input,
        n_in=32*4*4,
        n_out=4096,
        W=None,
        b=None,
        activation=theano.tensor.nnet.relu,
        p=0.7
    )
    
    
    
    
    # construct a fully-connected sigmoidal layer
    layer6 = DropoutHiddenLayer(
        is_train= training_enabled,
        rng=rng,
        input=layer5.output,
        n_in=4096,
        n_out=512,
        W=None,
        b=None,
        activation=theano.tensor.nnet.relu,
        p=0.7
    )
    layer62 = DropoutHiddenLayer(
        
        is_train= training_enabled,
        rng=rng,
        input=layer6.output,
        n_in=512,
        n_out=512,
        W=None,
        b=None,
        activation=theano.tensor.nnet.relu,
        p=0.7
    )
    L2_reg=0.0001
    
    # classify the values of the fully-connected sigmoidal layer
    layer7 = LogisticRegression(input=layer62.output, n_in=512, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer7.negative_log_likelihood(y)

    L2_sqr = (
        
         (layer7.W ** 2).sum()
    )
    cost = cost# + L2_sqr
    
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer7.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            training_enabled: numpy.cast['int32'](0)
        }
    )

    validate_model = theano.function(
        [index],
        layer7.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
            training_enabled: numpy.cast['int32'](0)
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer62.params + layer6.params + layer7.params + layer5.params  + layer3.params + layer1.params + layer0.params + layer01.params #+ layer02.params + layer03.params+ layer04.params + layer05.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    
   
    
    """
The MIT License (MIT)

Copyright (c) 2015 Alec Radford

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""    

    def Adam(cost, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
        updates = []
        grads = T.grad(cost, params)
        i = theano.shared(floatX(0.))
        i_t = i + 1.
        fix1 = 1. - (1. - b1)**i_t
        fix2 = 1. - (1. - b2)**i_t
        lr_t = lr * (T.sqrt(fix2) / fix1)
        for p, g in zip(params, grads):
            m = theano.shared(p.get_value() * 0.)
            v = theano.shared(p.get_value() * 0.)
            m_t = (b1 * g) + ((1. - b1) * m)
            v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
            g_t = m_t / (T.sqrt(v_t) + e)
            p_t = p - (lr_t * g_t)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t))
        updates.append((i, i_t))
        return updates

    updates = Adam(cost, params)
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            training_enabled: numpy.cast['int32'](1)
        }
        
    )
    train_model_augmented = theano.function(
        [mydata, index],
        
        cost,
        updates=updates,
        givens={
            x: mydata,
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            training_enabled: numpy.cast['int32'](1)
        }
        
    )
    # end-snippet-1
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (not done_looping):
        epoch = epoch + 1
        if(epoch>=50):
            break
        '''
        train_set[0] = noise_image(train_set[0], 0.05)
        train_set[0] = flip_image(train_set[0],0.5)
        train_set[0] = rotate_image(train_set[0],0.15)
        train_set[0] = translate_image(train_set[0],0.15)
        
        train_set_x, train_set_y = shared_dataset(train_set)
        '''
        for minibatch_index in range(n_train_batches):
            
            iter = (epoch - 1) * n_train_batches + minibatch_index
        
            if iter % 100 == 0:
                print('training @ iter = ', iter)
            
            temp_data = train_set_x.get_value() 
            data = temp_data[minibatch_index * batch_size:  (minibatch_index+1) *batch_size]
            data = noise_image(data, 0.05)
            data = rotate_image(data, 0.15)
            data = translate_image(data, 0.25)
            data = flip_image(data,0.5)
            #mydata.set_value(data)
            cost_ij = train_model_augmented(data, minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))
    #train_set = numpy.asarray(train_set)
    #valid_set = numpy.asarray(valid_set)
    #print(numpy.shape(train_set))
    #print(numpy.shape(valid_set))
    #train_set.append(valid_set)
    #print(train_set.shape())
    #train_set_x, train_set_y = shared_dataset(train_set)
    #n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    #n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    #n_train_batches //= batch_size
    '''
    epoch = 0
    print('...........................final testing on entire dataset..................')
    while (epoch <20) :
        epoch = epoch + 1
        train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            training_enabled: numpy.cast['int32'](1)
            }
        
        )
        train_set[0] = noise_image(train_set[0], 1)
        train_set[0] = flip_image(train_set[0],0.5)
        train_set[0] = rotate_image(train_set[0],1)
        train_set[0] = translate_image(train_set[0],1)
        
        train_set_x, train_set_y = shared_dataset(train_set)
        
        for minibatch_index in range(n_train_batches):
            
            iter = (epoch - 1) * n_train_batches + minibatch_index
        
            if iter % 100 == 0:
                print('training @ iter = ', iter)
           
            cost_ij = train_model(minibatch_index)
            
            test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
            test_score = numpy.mean(test_losses)
            print(('     epoch %i, minibatch %i/%i, test error of best model %f %%') %(epoch, minibatch_index + 1, n_train_batches, test_score * 100.))
         
                    
    

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('with test performance')
    print(test_score *100)
    print(('The code for file ' +
           
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))
    '''
test_mynet()
