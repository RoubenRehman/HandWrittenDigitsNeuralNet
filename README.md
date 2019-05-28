# RESOURCES
The MNIS dataset used in this small project consists of 60,000 training images
with labels and 10,000 examples with labels. It was created by

- Yann LeCun, Courant Institue, NYU<br>
- Corinna Cortes, Google Labs, New York<br>
- Christopher J.C. Burges, Microsoft Research, Redmond<br>

Yann LeCun's version of the dataset, the one used in this project, can be found here:

http://yann.lecun.com/exdb/mnist/

I do not own this dataset, nor did I contribute to it obviously. All the credit goes to the creators mentioned above.
I do not plan to earn money with it or use it in any comercial sense or form, this is very clearly only a private project to
get myselfe into the development machine learning.

# CODE
I used Gosh4AI's code to read the MNIST dataset, parse it into a numpy array and output the images as png files, if desired.
His video and GitHub can be found here:

https://www.youtube.com/watch?v=6xar6bxD80g&t=1782s
https://github.com/Ghosh4AI


For the neural net, I modified jonasbostoen's code (aka polycode) to fit my needs. His original video and GitHub can be found here:

https://www.youtube.com/watch?v=kft1AJ9WVDk
https://github.com/jonasbostoen/simple-neural-network

# ARCHITECTURE

My net uses one input layer with 784 neurons, each corresponding to one pixel of the image fed into it. It utilizes two hidden layers with 16 neurons each to preocess the data and activates an output layer of 9 neurons accordingly. Each neuron of the output layer corresponds to one of the nine digits, so neuron 0 is 0, neuron 1 is 1 and so on. The layers are represented by numpy arrays, so in the output layer, an index represents a digit and the value of the array at that index indicates, how sure the net 'thinks' that this digit was read from the picture in percent. Later, i plan on implementing an interface that allows the user to change and alter this architecture.
