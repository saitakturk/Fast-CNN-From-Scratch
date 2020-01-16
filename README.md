# Fast-CNN-From-Scratch

This repo contains convolutional neural network layers that are 2D convolution, dropout, maxpooling, flatten, dense. I implemented forward propagation and backpropagation of each layer type from scratch. I used PyTorch library in general for mathematical calculations and convolution operation( because of time complexity of convolution operation, I used directly Conv2D from PyTorch, both forward propagation and backpropagation). For the optimizer, I implemented SGD with momentum.

## Convolution Layer

The default kernel size 3 and stride 1.( I will update this part, however, in practice, you can achieve same result with stacking multiple convolution layer to reach bigger kernel size, and it is also computationally efficient way to reach bigger kernel sizes.) For back propagation you can see the calculations in below:

![Convolution Layer Backpropagation Formula](https://miro.medium.com/max/800/1*JFoSzff2lxlWn8nJjGZcMw.png)

## MaxPooling Layer

I used PYTorch MaxPool2D Layer for forward propagation and for backpropagation MaxUnPool2D Layer with saving information about locations of max response.

![](https://www.researchgate.net/profile/Eli_David/publication/306081538/figure/fig2/AS:418518853013507@1476794078414/Pooling-and-unpooling-layers-For-each-pooling-layer-the-max-locations-are-stored-These.png)

## Dropout Layer

I implemented dropout layer dense layers that is basically applies the same random binomial tensor with given range of dropout both forward propagation and backpropagation.

![](https://dpzbhybb2pdcj.cloudfront.net/allaire/v-1/Figures/dropout.png)


## Flatten Layer

I implemented by reshaping 4D tensor to 2D tensor.

## Dense Layer( aka Fully Connected Layer )

Dense layer is basically simple MLP layer that uses the my ReLU implementation as an activation function. 

![](https://miro.medium.com/max/1394/1*Np-nCh7o6JSs7Vj06BF9ag.png)
![](https://miro.medium.com/max/1144/1*1qbbBLUjAmoR3BkUqt_F9Q.png)

## Output Layer ( Softmax Layer ) 

This layer applies optimized softmax activation function and calculates categorical cross entropy both forward and backpropagation.

![](https://miro.medium.com/max/3528/1*R4uaiqeO517WJ3fb2yawLw.png)
