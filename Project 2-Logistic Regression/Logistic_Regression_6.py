# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 13:46:29 2018

@author: Ryan Jaipersaud
https://archive.ics.uci.edu/ml/datasets/LSVT+Voice+Rehabilitation
This method predicts whether voice rehabilitation treatment lead to phonations 
considered 'acceptable' or 'unacceptable' (binary class classification problem).


"""
import csv
import numpy
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

class Sigmoid():
    # this function outputs sigmoid values from 0 to 1 to a vector for a given X (observation) and theta
    def sigmoid_vector(self,X,theta):
        i = 0
        H = [] # resets H after each run
        for row in X: # for each observation
            temp = numpy.matmul(theta.T,X[i,:]) # multiplies theta by each observation
            H = numpy.append(H,1/(1+numpy.exp(-temp))) # fill in sigmoid vector H = 1/(1+e^(theta_T*x)) for that row/observation
            i = i+1
        H = numpy.array([H])  
        H = H.T
        return H
    # this function returns the number of correct prediction and modifies the sigmoid vector
    def predicted_output(self,Y,H):
        i = 0
        correct = 0
        for element in H:
            if  H[i,0] <= 0.5:
                H[i,0] = 0
            else:
                H[i, 0] = 1
            if H[i,0] == Y[i,0]:
                correct = correct + 1
            i = i + 1
        return correct

FileDestination1 = 'LSVT_data.csv'
FileDestination2 = 'LSVT_response.csv'

with open(FileDestination1) as csv_file1:
    csv_reader = csv.reader(csv_file1, delimiter=',')
    X = []
    i = 0
    for row in csv_reader:
        if i == 0:
            i = i + 1
            continue # skips the header row
        X.append(row) # adds the row to X

X = numpy.array(X) #turns X into a numpy object
X = X.astype(numpy.float) # converts strings to floats
X = X[:,0:9]

with open(FileDestination2) as csv_file2:
    csv_reader = csv.reader(csv_file2, delimiter=',')
    Y = []
    i = 0
    for row in csv_reader:
        if i == 0:
            i = i + 1
            continue # skips the header row
        Y.append(row) # adds the row to X

Y = numpy.array(Y) #turns X into a numpy object
Y = Y.astype(numpy.float) # converts strings to floats
Y = Y - 1 # makes Y live in the set [0,1]

# Now You have X and Y processed an ready to use 

X0 = numpy.ones((X.shape[0],1))
X = numpy.hstack((X0,X)) # add a one vector to the beginning of X 
X = normalize(X,axis = 0) # normalize each column



# Create training data
Xtrain = numpy.array(X[0:80,:]) 
Xtrain = numpy.array(Xtrain)
Ytrain = numpy.array(Y[0:80,0])
Ytrain = numpy.array(numpy.transpose([Ytrain]))

# Create test data
Xtest = numpy.array(X[81:125,:])
Ytest = numpy.array(Y[81:125,0])
Ytest = numpy.array(numpy.transpose([Ytest]))


Y0 = numpy.ones((Ytrain.shape[0],1)) # one vector later used for subtraction
H0 = numpy.ones((Ytrain.shape[0],1))  # one vector later used for subtraction
#theta = numpy.full((X.shape[1], 1),0) # initial guess for SGD
theta = numpy.zeros((X.shape[1], 1))
theta2 = numpy.zeros((Xtrain.shape[1], 1)) # initial guess for theta regularized
#print(numpy.matmul(theta.transpose(),Xtrain[i,:].transpose()))


Log_Likelyhood_funct_1 = numpy.array([])
Log_Likelyhood_funct_2 = numpy.array([])
iterations = numpy.array([])


for z in range(40): # iterations for theta to converge
    
    S = Sigmoid()
    H = S.sigmoid_vector(Xtrain,theta) #unregularized
    H2 = S.sigmoid_vector(Xtrain,theta2) # regularized
    
    # unregularized case
    YPrime = numpy.subtract(Y0,Ytrain) # YPrime is 1 - Y
    HPrime = numpy.subtract(H0,H) # HPrime is 1 - H
    Log_Likelyhood = numpy.matmul(Ytrain.T,numpy.log(H)) + numpy.matmul(YPrime.T,numpy.log(HPrime)) 
    Log_Likelyhood_funct_1 = numpy.append(Log_Likelyhood_funct_1,Log_Likelyhood) #  = Transpose(Y)*log(H) + Transpose(1-Y)*log(1-H)
    
    #regularized case
    HPrime2 = numpy.subtract(H0,H2) # HPrime is 1 - H
    Log_Likelyhood2 = numpy.matmul(Ytrain.T,numpy.log(H2)) + numpy.matmul(YPrime.T,numpy.log(HPrime2)) 
    Log_Likelyhood_funct_2 = numpy.append(Log_Likelyhood_funct_2,Log_Likelyhood2)
    
    iterations = numpy.append(iterations,z+1)
    for i in range(Xtrain.shape[0]): # update theta over all observations
        alpha = 0.1 
        L = 0.001 # lambda value
        for j in range(Xtrain.shape[1]): # update j over all features
            theta[j,0] = theta[j,0] + alpha*(Ytrain[i,0]-H[i,0]) * Xtrain[i,j] # update theta for a specific observation i
            theta2[j,0] = theta2[j,0] + alpha*(Ytrain[i,0]-H2[i,0]) * Xtrain[i,j] - L*theta2[j,0] # update theta for a specific observation i
            

print('The theta for the unregularized case is')
print(theta)
Htest = S.sigmoid_vector(Xtest,theta)
correct = S.predicted_output(Ytest,Htest)
#print(Htest)
print('The % correct for the unregularized case is ', 100*correct/Ytest.shape[0])

print('The theta for the regularized case is')
print(theta2)
Htest2 = S.sigmoid_vector(Xtest,theta2)
correct = S.predicted_output(Ytest,Htest2)
#print(Htest)
print('The % correct for the regularized case is ', 100*correct/Ytest.shape[0])



plt.plot(iterations,Log_Likelyhood_funct_1, 'r', iterations, Log_Likelyhood_funct_2, 'b' )
plt.xlablel =('iterations')
plt.ylablel =('Log Likelyhood')
plt.title('Log Likelyhood Convergence')
plt.legend()
plt.show()