#programming assignment 1
import matplotlib.pyplot as plt
import numpy as np
import sklearn

trainaccuracy=[]
testaccuracy=[]
input= 785
output = 10
training = 50000
test= 8000
layers = [20, 50, 100]
momentum= [0, 0.25, 0.5]
learning = 0.1
runs=50

# Converting into csv

def convert(img, label, out, n):
    f = open(img, "rb")
    o = open(out, "w")
    l = open(label, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28 * 28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image) + "\n")
    f.close()
    o.close()
    l.close()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def derivativesigmoid(z):
    return z * (1 - z)

def loadfile(fileName):
    dataFile = np.loadtxt(fileName, delimiter=',')
    inputvalue = np.insert(dataFile[:, np.arange(1, inputSize)] / 255, 0, 1, axis=1)
    outputvalue = dataFile[:, 0]
    return inputvalue, outputvalue
# Converting sets to csv
print("\nConverting Training Set")
convert("train-images.idx3-ubyte", "train-labels.idx1-ubyte",
        "mnist_train.csv", 50000)
print("\nConverting Testing Set")
convert("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte",
        "mnist_test.csv", 8000)
# Load Training and Test Sets :
print("\nLoading Training Set")
traindata, traininglabels = loadfile('mnist_train.csv')
print("\nLoading Test Set\n")
testdata, testlabel = loadfile('mnist_test.csv')
print("\nExperiment 1")
for nhiddenlayer in layers:
    neuralnet(nhiddenlayer, 0.9)
print("\nExperiment 2 ")
for momentum in momentumValues:
    neuralnet(100, momentum)

print("\nExperiment 3 ")
for i in range(0, 2):
    traindata, X, traininglabels, Y = train_test_split(traindata, traininglabels, test_size=0.50)
    training = int(training / 2)
    neuralnet(100, 0.9)		
# Output after applying sigmoid function to the layers with weights by forward propagation
def fprop(dataSet, jinput, joutout):
    inputactivation= np.reshape(dataSet, (1, inputSize))
    hiddenactivation = sigmoid(np.dot(inputactivationlayer, jinput))
    hiddenactivation[0][0] = 1
    outputactivation= sigmoid(np.dot(hiddenactivation, joutout))
    return inputactivationlayer, hiddenactivation, outputactivation

# Calculating errors, delta, momentum and using old weight to new weights 
def backprop(error, inputactivationlayer, hiddenactivation, outputactivation,
                    joutout, jinput, joutout_oldValues,
                    jinput_oldValues, momentum):
    deltaout = derivativesigmoid(outputactivation) * error
    deltahidden = derivativesigmoid(hiddenactivation) * np.dot(deltaout,
                                                                                           np.transpose(
                                                                                               joutout))
    wihiddenout = (learning * np.dot(np.transpose(hiddenactivation), deltaout)) + (
                momentum * joutout_oldValues)
    whiddeninout = (learning * np.dot(np.transpose(inputactivationlayer), deltahidden)) + (
                momentum * jinput_oldValues)
    joutout += wihiddenout
    jinput += whiddeninout
    return joutout, jinput, wihiddenout, whiddeninout

# Training network
def train(joutout, jinput, joutout_oldValues,
                       jinput_oldValues, momentum):
    for i in range(0, training):
        # forward prop
        inputactivationlayer, hiddenactivation, outputactivation= fprop(traindata[i, :],
                                                                                                jinput,
                                                                                                joutout)
        # target value calculation for this input.
        targetOutput = np.insert((np.zeros((1, output - 1)) + 0.001), int(traininglabels[i]), 0.999)
        # Calculate updated weight matrix after backpropagating errors.
        joutout, jinput, joutout_oldValues, jinput_oldValues = backprop(
            targetOutput - outputactivation, inputactivationlayer, hiddenactivation, outputactivation,
            joutout, jinput, joutout_oldValues,
            jinput_oldValues, momentum)
    return jinput, joutout
# Testing network
def test(inputvalue, outputvalue, sizeOfSet, jinput, joutout):
    predictout = []
    for i in range(0, sizeOfSet):
        # forward propagating for calculating activations at each layer.
        inputactivationlayer, hiddenactivation, outputactivation= fprop(inputvalue[i, :],
                                                                                                jinput,
                                                                                                joutout)
       
        predictout.append(np.argmax(outputactivation))
    return accuracy_score(outputvalue, predictout), predictout
def neuralnet(nhiddenlayer, momentum):

    # assigning weights
    jinput = (np.random.rand(inputSize, nhiddenlayer) - 0.5) * 0.1
    joutout = (np.random.rand(nhiddenlayer, output) - 0.5) * 0.1
    # assigning all previous weights to 0
    joutout_oldValues = np.zeros(joutout.shape)
    jinput_oldValues = np.zeros(jinput.shape)
    # to run for 70 epochs	
    for epoch in range(0,runs):
        # Calculate the training accuracy and output values.
        trainacc, predictout = test(traindata, traininglabels, training,
                                                              jinput, joutout)
		trainingaccuracy.append(trainacc)
        # Calculate the testing accuracy and output values.
        testacc, predictout = test(testdata, testlabel, tsize,
                                                             jinput, joutout)
		testaccuracy.append(testacc)
        print("Epoch " + str(epoch) + " :\tTraining Set Accuracy = " + str(
            trainacc) + "\n\t\tTest Set Accuracy = " + str(testacc))
        # computing new weights by training after each epoch
        jinput, joutout = train(joutout,
                                                                                 jinput,
                                                                                 joutout_oldValues,
                                                                                 jinput_oldValues,
                                                                                 momentum)
	plt.plot(trainaccuracy)
	plt.plot(testaccuracy)
    epoch += 1
    # Calculate the final training accuracy and output values.
    trainacc, predictout = test(traindata, traininglabels, training,
                                                          jinput, joutout)
    # Calculate the final testing accuracy and output values.
    testacc, predictout = test(testdata, testlabel, tsize,
                                                         jinput, joutout)
    print("Epoch " + str(epoch) + " : Accuracy = " + str(
        trainacc) + "\n\ Accuracy = " + str(testacc) + "Hidden Layer Size = " + str(
        nhiddenlayer) + "\tMomentum = " + str(momentum) + "Training Samples = " + str(
        training) + "\n\nConfusion Matrix :\n")
    print(confusion(testlabel, predictout))
    print("\n")
    return


