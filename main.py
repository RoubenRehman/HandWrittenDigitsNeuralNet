# Imports
import os
import codecs
from skimage.io import imsave
import numpy as np


# Squashing function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# testing function
def test_net():
    for it in range(10000):

        # feed forward
        l0 = data_dict['test_images'][it].reshape((1, 784))
        l1 = sigmoid(np.dot(l0, syn0))
        l2 = sigmoid(np.dot(l1, syn1))
        l3 = sigmoid(np.dot(l2, syn2))

        percentage = np.amax(l3)

        for i in range(l3.size):
            if l3[0][i] == percentage:
                guessed_number = i
                break

        print("Label: " + str(data_dict['test_labels'][it]))
        print("Guessed: " + str(guessed_number) + " with a percentage of " + str(percentage * 100))

        ans = input();
        if ans == "exit":
            break

        else:
            continue


# path to resource data
datapath = './resources/'
files = os.listdir(datapath)


# function to determine int value from read hex
def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


data_dict = {}
for file in files:
    if file.endswith('ubyte'):
        print('Reading ', file)
        with open(datapath+file, 'rb') as f:
            data = f.read()
            type = get_int(data[:4])        # reading magic number
            length = get_int(data[4:8])     # length of data array
            if type == 2051:    # magic number indicates images
                category = 'images'
                num_rows = get_int(data[8:12])
                num_cols = get_int(data[12:16])
                parsed = np.frombuffer(data, dtype = np.uint8, offset = 16)
                parsed = parsed.reshape(length, num_rows, num_cols)

            elif type == 2049:  # magic number indicates labels
                category = 'labels'
                parsed = np.frombuffer(data, dtype = np.uint8, offset = 8)
                parsed = parsed.reshape(length)

            if length == 10000:     # testing set
                set = 'test'

            elif length == 60000:   # training set
                set = 'train'

            data_dict[set+'_'+category] = parsed

# outputting the numpy arrays as png pictures
query = "blank"

# test_output for array structure
# x = data_dict['test_images'][3]
# print(x)

while query != "y" and query != "n":
    query = input("Do you want to output the read images as png (y/n)?\n")

if query == "y":
    sets = ['train', 'test']

    for set in sets:
        images = data_dict[set+'_images']
        labels = data_dict[set+'_labels']
        no_of_samples = images.shape[0]

        for i in range(no_of_samples):
            print(set, i)
            image = images[i]
            label = labels[i]
            if not os.path.exists(datapath+set+'/'+str(label)+'/'):
                os.makedirs(datapath+set+'/'+str(label)+'/')

            filenumber = len(os.listdir(datapath+set+'/'+str(label)+'/'))
            imsave(datapath+set+'/'+str(label)+'/%05d.png'%filenumber, image)

else:
    print("No output generated\n")

# seed
seed = input("Enter a seed for the random gegnerator!")
np.random.seed(int(seed))

# Planned structure
# Input Layer with
#
# + ---- + ---- + ...
# + ---- + ---- + ...
# + ---- + ---- + ...
# + ---- + ---- + ... ---- +
# . ---- + ---- + ... ---- +
# .. --- + ---- + ... ---- +
# .. --- + ---- + ... ---- +
# ... -- + ---- + ... ---- +
# .. --- + ---- + ... ---- +
# .. --- + ---- + ... ---- +
# . ---- + ---- + ... ---- +
# + ---- + ---- + ... ---- +
# + ---- + ---- + ...
# + ---- + ---- + ...
# + ---- + ---- + ...
# + ---- + ---- + ...

# synapses 784
syn0 = 2*np.random.random((784, 16)) - 1
syn1 = 2*np.random.random((16, 16)) - 1
syn2 = 2*np.random.random((16, 10)) - 1

# defining input layer
l0 = np.empty((1, 784))

# normalizing values
data_dict['train_images'] = data_dict['train_images'] / 255
data_dict['test_images'] = data_dict['test_images'] / 255

ans = input("\nSetup complete, do you want to test the net on the testing images before training (y/n)?\n")
while ans != "y" and ans != "n":
    ans = input("Setup complete, do you want to test the net on the testing images before training (y/n)?")

if ans == "y":
    print("Entering testing mode, type 'exit' to continue to training mode\n\n")
    test_net()

else:
    print("\n\n")

# Training
for it in range(60000):

    # feed forward
    l0 = data_dict['train_images'][it].reshape((1, 784))
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))
    l3 = sigmoid(np.dot(l2, syn2))

    # back propagation
    expt_out = np.zeros((1, 10))
    expt_out[0][data_dict['train_labels'][it]] = 1
    l3_err = expt_out - l3

    # calculating deltas
    l3_delta = l3_err*sigmoid_derivative(l3)

    l2_err = l3_delta.dot(syn2.T)
    l2_delta = l2_err*sigmoid_derivative(l2)

    l1_err = l2_delta.dot(syn1.T)
    l1_delta = l1_err*sigmoid_derivative(l1)

    # updating
    syn2 += l2.T.dot(l3_delta)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

    if (it % 10000) == 0:
        print("training...")

# print("Output after training")
# print("Label: " + str(data_dict['train_labels'][59999]))
# print(l3*100)

print("\n\nTraining completed, showing result on test samples. Press enter to continue, type 'exit' to exit!\n")
test_net()

