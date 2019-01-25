import numpy as np
import torch
import torch.nn as nn
import torch.cuda as tc
'''
To run this code the tester should install only two libraries that numpy and torch 
-Install libraries with( You should first install Anaconda )  
*conda install pytorch torchvision -c pytorch
*pip install numpy


'''

'''
We are using cifar100 datasets to test our CNN implementation. Therefore to test our implementation 
you should install tensorflow and keras for only getting cifar100 dataset.
-Install libraries with 
*pip install tensorflow
*pip install keras

'''
from keras.datasets import  cifar100


class Conv:
    '''
    This class applies conv2d operation using torch cuda tensors
    '''

    def __init__(self, input_of_layer, kernel_number, kernel_size=3, stride_size=1):
        '''
        This function create Conv instance wtih given parameters
        :param input_of_layer: input of convolution layer
        :param kernel_number: number of kernel that will be applied
        :param kernel_size: kernel size for convolution operation
        :param stride_size: stride size for convolution operation
        '''
        # initialize parameters
        self.input = tc.FloatTensor(input_of_layer)
        if len(self.input.shape) != 4:
            raise ValueError('Input size is ' + len(
                self.input.shape) + " Should at least 4 dimension (batch_size, channel_size, width, height")
        self.kernel_number = kernel_number
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        # get dimensions
        self.batch_size = self.input.shape[0]
        self.channel_size = self.input.shape[1]
        self.width_size = self.input.shape[2]
        self.height_size = self.input.shape[3]
        # conv2d operation for forward pass, we are using torch.Conv2d for only fast speed
        self.layer = nn.Conv2d(self.channel_size, self.kernel_number, self.kernel_size, self.stride_size,
                               self.padding_size()).cuda().type(torch.float32)
        # initialized weights and biases wtih xavier initializiation
        nn.init.xavier_uniform_(self.layer.weight)

        # terms for momentum saving
        self.delta_w = 0
        self.delta_b = 0

    def forward(self, input=tc.FloatTensor(0)):
        '''
        This function runs forward propagation for Convolution operation
        :param input: output of previous layer
        :return: output of convolution of input with relu activation function
        '''
        self.input = input
        self.act_output = self.relu(self.layer.forward(tc.FloatTensor(self.input)))
        return self.act_output

    def padding_size(self):
        '''
        This function finds padding size for maintain sizess
        :return: return padding size
        '''
        return (self.kernel_size - 1) // 2

    def relu(self, xa, derive=False):
        '''
        This function applies relu forward and back propagation via determining derive parameter
        :param xa: output of previous layer
        :param derive: if it is True, the function runs backprop for relu, otherwise, it runs forwardprop
        :return: according derive parameter, it change but give relu operation results
        '''
        # back prop
        if derive:
            return torch.ceil(torch.clamp(xa, min=0, max=1)).detach()
        # forward prop
        return torch.clamp(xa, min=0).detach()

    def backward(self, lr_rate, momentum_rate, next_layer_grad, isInput=False):
        '''
        This function runs backprop for convolution layer
        :param lr_rate: learning rate for weight updates
        :param momentum_rate: momentum rate for momentum operations
        :param next_layer_grad: gradients from next layer
        :param isInput: check whether input of the layer is input, if it is, then there is no need for gradient calculation
        otherwise, calculate input gradient
        :return: gradient of the input
        '''
        # found gradient of activation function operation
        self.act_output = torch.mul(next_layer_grad, self.relu(self.act_output, derive=True)).detach()

        # this convolution operation for calculating gradient of weights
        dw_layer = nn.Conv2d(self.batch_size, self.kernel_number, self.act_output.shape[2], 1,
                             self.padding_size()).cuda().type(torch.float32)
        # copying activation gradient
        gradx = self.act_output.clone().detach()
        # flipping for convolution operation
        gradx = gradx.cpu().detach().numpy()[:, :, ::-1, ::-1]
        # set weight and bias with 0  with activation gradient
        dw_layer.weight.data = tc.FloatTensor(gradx.copy()).transpose(0, 1).detach()
        del gradx
        dw_layer.bias.data = tc.FloatTensor(np.zeros(self.kernel_number)).detach()
        # momentum update
        self.delta_w = momentum_rate * self.delta_w + (
                dw_layer.forward(self.input.transpose(0, 1)).transpose(0, 1).detach() / (self.batch_size)).detach()
        self.delta_b = momentum_rate * self.delta_b + (torch.sum(self.act_output, dim=[0, 2, 3]).detach() / (
                self.act_output.shape[0] * self.batch_size * self.act_output.shape[2] * self.act_output.shape[
            3])).detach()
        # weight and bias update
        self.layer.weight.data -= lr_rate * self.delta_w.detach()
        self.layer.bias.data -= lr_rate * self.delta_b.detach()

        # if it is not input that initially given model then calculate input gradient as well
        if (isInput == False):
            # convolution operation for calculating input gradient
            dx_layer = nn.Conv2d(self.kernel_number, self.channel_size, self.kernel_size, self.stride_size,
                                 self.padding_size()).cuda().type(torch.float32)
            # get weight from conv layer
            temp_weight = self.layer.weight.data.clone().cpu().numpy()
            # flip kernel for convolution operation
            temp_weightx = temp_weight[:, :, ::-1, ::-1]
            #set weight with flipped kernel
            dx_layer.weight.data = tc.FloatTensor(temp_weightx.copy()).transpose(0, 1).detach()
            #set bias with zero
            dx_layer.bias.data = tc.FloatTensor(np.zeros(self.channel_size)).detach()
            #input gradient
            out = dx_layer.forward(self.act_output)

            del temp_weight
            del temp_weightx

            return out
        else:
            #return input itself
            # del self.act_output
            return self.input


class MaxPooling:
    '''
    This class applies maxpooling operation
    '''
    def __init__(self, input, kernel_size=2, stride_size=2):
        '''
        This function create instance for maxpooling layer
        :param input: output of previous layer
        :param kernel_size: to apply max pooling
        :param stride_size: stride size for max pooling
        '''
        self.input = input.detach()
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        #this applies max pooling
        self.layer = nn.MaxPool2d(self.kernel_size, self.stride_size, return_indices=True).cuda().type(torch.float32)

    def forward(self, input=tc.FloatTensor(0)):
        '''
        This function runs forward propagation of maxpooling
        :param input: output of previous layer
        :return: max pooling operation result
        '''

        self.input = input.detach()
        self.output, self.indices = self.layer.forward(self.input)
        return self.output

    def backward(self, next_layer_grad):
        '''
        This function runs backprop for maxpooling
        :param next_layer_grad: gradient of next layer
        :return: backprop of max pooling
        '''
        #unpool operation for back propagation
        back_layer = nn.MaxUnpool2d(self.kernel_size, self.stride_size).cuda().type(torch.float32)
        #output of backpropagation
        out = back_layer.forward(next_layer_grad, self.indices, output_size=self.input.shape).type(
            torch.float32).detach()
        del back_layer

        return out


class Dense:
    '''
    This class applies fully connected layer
    '''

    def __init__(self, input, dense_number):
        '''
        This function creates dense instance with given dense size
        :param input: output of previous layer
        :param dense_number: number of dense layer
        '''

        self.input = input.detach()
        self.dense_number = dense_number

        self.batch_size = self.input.shape[0]
        self.input_size = self.input.shape[1]
        #initialize weights and biases
        self.weight = tc.FloatTensor(
            np.random.normal(size=(self.input_size, self.dense_number), loc=0, scale=0.01).astype(np.float32)).type(
            torch.float32).detach()
        self.bias = tc.FloatTensor(np.zeros((1, dense_number))).type(torch.float32).detach()
        #momentum terms
        self.delta_w = 0
        self.delta_b = 0

    def forward(self, input=tc.FloatTensor(0).detach()):
        '''
        This function runs forward prop fully connected layer
        :param input: the previous layer of output
        :return: the activation function output applied dense operation
        '''

        self.input = input.detach()
        self.act_output = self.relu(torch.mm(self.input, self.weight) + self.bias).detach()
        return self.act_output

    def relu(self, xa, derive=False):
        '''
        This function applies relu forward and back propagation via determining derive parameter
        :param xa: output of previous layer
        :param derive: if it is True, the function runs backprop for relu, otherwise, it runs forwardprop
        :return: according derive parameter, it change but give relu operation results
        '''
        # back prop
        if derive:
            return torch.ceil(torch.clamp(xa, min=0, max=1)).detach()
        # forward prop
        return torch.clamp(xa, min=0).detach()

    def backward(self, lr_rate, momentum_rate, next_layer_grad, isInput=False):
        '''
        This function runs back prop of fully connected layer
        :param lr_rate: learning rate
        :param momentum_rate: momentum rate
        :param next_layer_grad: gradient from next layer
        :param isInput: check whether input is data
        :return: return input gradient
        '''
        #find gradient activation functions
        self.act_output = torch.mul(next_layer_grad, self.relu(self.act_output, derive=True)).type(
            torch.float32).detach()
        #find momentum terms
        self.delta_w = momentum_rate * self.delta_w + (
                torch.mm(self.input.transpose(0, 1), self.act_output).detach() / self.batch_size).detach()
        self.delta_b = momentum_rate * self.delta_b + (
                torch.sum(self.act_output, dim=0, keepdim=True).detach() / self.batch_size).detach()
        #update weight and bias
        self.weight -= lr_rate * self.delta_w.detach()
        self.bias -= lr_rate * self.delta_b.detach()
        #check whether input is data, if it is not, then find input gradient
        if isInput == False:
            #gradient operation for input
            out = torch.mm(self.act_output, self.weight.transpose(0, 1)).type(torch.float32).detach()
            return out
        else:
            #return input itself
            return self.input


class Flatten:
    '''
    This classes flattens the data from prevoius layer
    '''
    def __init__(self, input):
        '''
        This function creates Flatten instance
        :param input: the out put of previous layer
        '''
        #set input and sizes
        self.input = input.detach()
        self.batch_size = self.input.shape[0]
        self.channel_size = self.input.shape[1]
        self.width = self.input.shape[2]
        self.height = self.input.shape[3]

    def forward(self, input=tc.FloatTensor(0).detach()):
        '''
        This function runs forward prop flatten operation from 4d  to 2d
        :param input: the previous layer output
        :return: 2d matrix
        '''
        self.input = input.detach()
        self.batch_size = self.input.shape[0]
        self.channel_size = self.input.shape[1]
        self.width = self.input.shape[2]
        self.height = self.input.shape[3]

        self.output = self.input.view(self.batch_size, -1).detach()
        return self.output

    def backward(self, next_layer_grad):
        '''
        This function runs backprop with reshaping 2d matrix to 4d tensor
        :param next_layer_grad: gradient from next layer
        :return: 4d tensor
        '''
        return next_layer_grad.view(self.batch_size, self.channel_size, self.width, self.height).detach()


class Dropout:
    '''
    This class applies dropout to fully connected layer
    '''
    def __init__(self, input, dropout):
        '''
        This function create instance for Dropout layer
        :param input: the output from previous layer
        :param dropout: dropout layer probability rate
        '''
        #check the input that has dimension of two
        if len(input.shape) != 2:
            raise ValueError('Input shape should be 2 dimensional')
        #initialize class attributes
        self.input = input.detach()
        self.dropout = dropout
        self.batch_size = self.input.shape[0]
        self.kernel_number = self.input.shape[1]
        #create dropout matrix
        self.dropout_matrix = tc.FloatTensor(np.zeros(self.input.shape)).detach()

    def forward(self, input):
        '''
        This function runs forward prop for dropout
        :param input: the output of the previous layer
        :return: dropout result
        '''
        self.input = input.detach()
        #create mask
        self.dropout_matrix = tc.FloatTensor(
            np.random.binomial(1, 1 - self.dropout, size=self.kernel_number)).expand_as(
            self.input).detach()
        #apply mask
        self.output = torch.mul(self.input, self.dropout_matrix).detach() / (1 - self.dropout + np.exp(-32))
        return self.output

    def backward(self, next_layer_grad):
        '''
        This function runs backprop of dropout layer
        :param next_layer_grad: the gradient of next layer
        :return: the gradient of the input
        '''
        return torch.mul(next_layer_grad, self.dropout_matrix).detach() / (1 - self.dropout + np.exp(-32))


class Output:
    '''
    This class applies categorical cross entropy with softmax activation function, it should be used last layer of the model
    '''
    def __init__(self, input, target):
        '''
        This function create instance for last layer of the model
        :param input: the output of previous layer
        :param target: target or labels
        '''
        #initialize attributes
        self.input = input.detach()
        self.target = target
        self.y = None
        self.class_number = self.input.shape[1]

    def forward(self, input=tc.FloatTensor(0).detach(), target=tc.FloatTensor(0).detach()):
        '''
        This function runs forward pass with softmax and categorical cross entropy
        :param input: the previous layer output
        :param target: the labels
        :return: the prediction
        '''
        self.input = input.detach()

        self.target = target
        self.output, self.loss = self.softmax_with_cross_entropy(self.input, self.target)

        return self.output

    def softmax_with_cross_entropy(self, x, target, derive=False):
        '''
        This function applies softmax activation function and categorical cross entropy
        :param x: the input
        :param target: label
        :param derive: if it is true it returns derivative of operation, otherwise it runs opeartion
        :return: according to derive parameters it runs either operation result or back prop result
        '''
        if derive == True:
            return x - target
        #softmax
        self.y = torch.eye(self.class_number)[tc.FloatTensor(target).type(torch.long).view(-1).detach(), :].cuda().type(
            torch.float32)

        maximum = torch.max(x, dim=1, keepdim=True)[0].detach()
        self.pred = torch.exp(x - maximum).detach()
        self.pred = self.pred.detach() / torch.sum(self.pred, 1, keepdim=True).detach()
        #categorical cross entropy
        self.loss = -torch.sum(self.y * torch.log(self.pred + np.exp(-32))).type(torch.float32).detach() / \
                    self.pred.shape[0]

        return self.pred, self.loss

    def backward(self):
        '''
        This function runs back prop for output layer
        :return: the gradient of input
        '''
        return self.softmax_with_cross_entropy(self.output, self.y, derive=True)

#load data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
#swap axes since it take 'channel_first'
x_part = x_train[:20].swapaxes(1, 3) / 255.0
#model
a01 = Conv(tc.FloatTensor(x_part).detach(), 32)
a01.forward(tc.FloatTensor(x_part).detach())
a001 = Conv(a01.act_output, 32)
a001.forward(a01.act_output)
a02 = MaxPooling(a001.act_output)
a02.forward(a001.act_output)
a1 = Flatten(a02.output)
a1.forward(a02.output)
a = Dense(a1.output, 512)
a.forward(a1.output)
a10 = Dropout(a.act_output, 0.05)
a10.forward(a.act_output)
b = Dense(a10.output, 128)
b.forward(a10.output)
b1 = Dropout(b.act_output, 0.15)
b1.forward(b.act_output)
c = Dense(b1.output, 100)
c.forward(b1.output)
d = Output(c.act_output, y_train[:20])
d.forward(c.act_output, y_train[:20])

import time
#epoch
for j in range(100):
    losses = []
    start = time.time()
    #minibatch
    for i in range(500):
        #forward pass
        x_part = x_train[i * 100:(i + 1) * 100].swapaxes(1, 3) / 255.0
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        a01.forward(tc.FloatTensor(x_part).detach())
        a001.forward(a01.act_output)
        a02.forward(a001.act_output)
        a1.forward(a02.output)
        a.forward(a1.output)
        a10.forward(a.act_output)
        b.forward(a10.output)
        b1.forward(b.act_output)
        c.forward(b1.output)
        d.forward(c.act_output, y_train[i * 100:(i + 1) * 100])
        losses.append(d.loss.cpu().numpy())
        #back pass
        e = c.backward(0.001, 0.90, d.backward().detach()).detach()
        e1 = b1.backward(e)
        f = b.backward(0.001, 0.90, e1).detach()
        f1 = a10.backward(f)
        g = a.backward(0.001, 0.90, f1, ).detach()
        h = a1.backward(g).detach()
        k = a02.backward(h).detach()
        k1 = a001.backward(0.001, 0.9, k).detach()
        m = a01.backward(0.001, 0.90, k1, True).detach()
        # del e, f, g, h, k, m
    end = time.time()
    print('Time',end - start)
    print(j, np.mean(losses))
