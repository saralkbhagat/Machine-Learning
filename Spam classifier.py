from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from numpy import genfromtxt


#Pre-processsing
print ("Reading data set")
data_set = genfromtxt('spambase.data', delimiter=',') 
data_length = len(data_set)
target = data_set[:, -1]
print ("Shuffling data set")
np.random.shuffle(data_set)
attributes_values = preprocessing.scale(data_set[:, 0:-1])
target = data_set[:, -1]
#splitting training and test data
X_train, X_test, y_train, y_test = train_test_split(
    attributes_values, target, test_size=0.5, random_state=17)

#Spam and not spam should be 40% and 60% respectively
totspam = 0
trainlen = len(X_train)
for i in range(trainlen):
    if y_train[i] == 1:
        totspam += 1		
spamprob = float(totspam) / trainlen
notspamprob = 1 - spamprob
print("The probability of the mail spam is: \t",spamprob)
print("The probability of mail not being spam is : \t",notspamprob)

#mean and SD for all the attributes_values
meanspam,SDspam,meannotspam,SDnonspam  = [], [],[],[]
for attributes_values in range(0,57):
    spam_values,nonspam_values = [],[]
    for i in range(0, trainlen):
        if (y_train[i] == 1):
            spam_values.append(X_train[i][attributes_values])
        else :
           nonspam_values.append(X_train[i][attributes_values])
    meanspam.append(np.mean(spam_values))
    meannotspam.append(np.mean(nonspam_values))
    SDspam.append(np.std(spam_values))
    SDnonspam.append(np.std(nonspam_values))
#replacing 0 standard deviation with .0001
for feature in range(0,57):
    if(SDspam[feature]==0):
        SDspam[feature] = .0001
    if(SDnonspam[feature]==0):
        SDnonspam[feature]=.0001
		
# This function is to calculate precision, Recall and accuracy.
def calculate(target, predicted, threshold):
    true_pos,false_pos,true_neg,false_neg = 0,0,0,0
    for i in range(len(predicted)):
        if (predicted[i] > threshold and target[i] == 1)  :
            true_pos += 1
        elif (predicted[i] > threshold and target[i] == 0 )  :
            false_pos += 1	
        elif (predicted[i] <= threshold and target[i] == 1 )  :
            false_neg += 1
        elif (predicted[i] <= threshold and target[i] == 0 )  :
            true_neg += 1
    accuracy = float(true_pos + true_neg) / len(predicted)
    recall = float(true_pos) / (true_pos + false_neg)
    precision = float(true_pos) / (true_pos + false_pos)
    return  accuracy, recall, precision
	
#Gaussian Naive Bayes Immplementation.
spamprobability,notspamprobability = 0,0
pred = []
for is in range(0,len(X_test)):
    spamNB,notspamNB = [],[]
    NB1,NB2,NB3,NB4 = 0,0,0,0
    for attributes_values in range(0,57):
        NB1 = float(1)/ (np.sqrt(2 * np.pi) * SDspam[attributes_values])
        NB2 = (np.e) ** - (((X_test[is][attributes_values] - meanspam[attributes_values]) ** 2) / (2 * SDspam[attributes_values] ** 2))
        spamNB.append(NB1 * NB2)
        NB3 = float(1)/ (np.sqrt(2 * np.pi) * SDnonspam[attributes_values])
        NB4 = (np.e) ** - (((X_test[is][attributes_values] - meannotspam[attributes_values]) ** 2) / (2 * SDnonspam[attributes_values] ** 2))
        notspamNB.append(NB3 * NB4)
		
    spamprobability = np.log(spamprob) + np.sum(np.log(np.asarray(spamNB)))
    notspamprobability = np.log(notspamprob) + np.sum(np.log(np.asarray(notspamNB)))
    output = np.argmax([notspamprobability, spamprobability])
    pred.append(output)
acc,rec,pre = calculate(y_test, pred, 0)
print("Confusion matrix:\n",metrics.confusion_matrix(y_test, pred))
print ("Accuracy Value: \t",acc)
print ("Precision Value: \t", pre)
print ("Recall Value: \t",rec)