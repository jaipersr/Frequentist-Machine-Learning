
"""
Created on Tue Oct 16 13:46:29 2018

@author: Ryan Jaipersaud
https://archive.ics.uci.edu/ml/datasets/LSVT+Voice+Rehabilitation
This method predicts whether voice rehabilitation treatment lead to phonations 
considered 'acceptable' or 'unacceptable' (binary class classification problem).
This methos is designed to show how to do cross validation the right way.


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
    def covariance(self,X,Y,Xtest):
        # for each feature in X determine the covariance of the feature with the output Y
        # for each X that gets reduced the Xtest must also be reduced to the same features
        features=numpy.ones((1,X.shape[1]))
        for j in range(X.shape[1]):
            column = numpy.array([X[:,j]]).T # something I had to do to get dimensions to work out
            COV = numpy.cov(column.T,Y.T)[0,1] # this take the 0,1 entry of the covariance matrix for the covariance of feature j with Y
            features[0,j] = COV
            index = numpy.argpartition(features, -5)[-5:] # returns index values for 5 highest covariances
        print('The most valuable features are',index[0,5:10])
        K = numpy.vstack( (X[:,index[0,5]], X[:,index[0,6]], X[:,index[0,7]], X[:,index[0,8]], X[:,index[0,9]]) ) # this reconstructs X to contain only the most important features
        K = K.T# K is now a 58 by 5
        
        K0 = numpy.vstack( (Xtest[:,index[0,5]], Xtest[:,index[0,6]], Xtest[:,index[0,7]], Xtest[:,index[0,8]], Xtest[:,index[0,9]]) ) # this reconstructs X to contain only the most important features
        K0 = K0.T
        
        return K, K0;
        

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

# Divide Training set in K sets where K = 4 in this case

K0 = Xtrain[0:19,:]
K1 = Xtrain[20:39,:]
K2 = Xtrain[40:59,:]
K3 = Xtrain[60:79,:]

Y0 = Ytrain[0:19,:]
Y1 = Ytrain[20:39,:]
Y2 = Ytrain[40:59,:]
Y3 = Ytrain[60:79,:]

for k in range(4): # iterations for theta to converge
    
    C = Sigmoid() # create a sigmoid object
    # Construct X and Y of the other K-1 folds
    if k == 0:
        X = numpy.vstack((K1,K2,K3))
        # Y0_p is the compliment of Y0 the p stands for prime
        Y0_p = numpy.vstack((Y1,Y2,Y3)) 
        # This find the (5) features in X with the greatest covariance with Y 
        # and return a matrix with only those features. The test data for the five features
        # will be unique to those five features the covariance selected so it must be updated as well
        X0, X0test = C.covariance(X,Y0_p,K0) 
        print('X0', X0.shape)
    elif k == 1:
        X = numpy.vstack((K0,K2,K3))
        Y1_p = numpy.vstack((Y0,Y2,Y3))
        X1, X1test = C.covariance(X,Y1_p,K1) # This find the features in X with the greatest covariance with Y and return a matrix with only those features
        print('X1',X1.shape)
    elif k == 2:
        X = numpy.vstack((K0,K1,K3))
        Y2_p = numpy.vstack((Y0,Y1,Y3))
        X2,X2test = C.covariance(X,Y2_p,K2) # This find the features in X with the greatest covariance with Y and return a matrix with only those features

        print('X2',X2.shape)
    elif k == 3:
        X = numpy.vstack((K0,K1,K2))
        Y3_p = numpy.vstack((Y0,Y1,Y2))
        X3, X3test = C.covariance(X,Y3_p,K3) # This find the features in X with the greatest covariance with Y and return a matrix with only those features
        print('X3',X3.shape)

print('------')

correctness = numpy.array([])
CVError = numpy.array([])
lambda_vector = numpy.array([])
L = 0.1 # lambda value
for y in range(10): # this is to generate different lambdas
    theta0 = numpy.zeros((X0.shape[1], 1))
    theta1 = numpy.zeros((X1.shape[1], 1))
    theta2 = numpy.zeros((X2.shape[1], 1))
    theta3 = numpy.zeros((X3.shape[1], 1))
    
    iterations = numpy.array([])
    for z in range(40): # iterations for theta to converge
        
        S = Sigmoid()
        H0 = S.sigmoid_vector(X0,theta0) 
        H1 = S.sigmoid_vector(X1,theta1) 
        H2 = S.sigmoid_vector(X2,theta2) 
        H3 = S.sigmoid_vector(X3,theta3)
        
        iterations = numpy.append(iterations,z+1)
        
        for i in range(X0.shape[0]): # update theta over all observations
            alpha = 0.1 
            
            for j in range(X0.shape[1]): # update j over all features
                theta0[j,0] = theta0[j,0] + alpha*(Y0_p[i,0]-H0[i,0]) * X0[i,j] - L*theta0[j,0] # update theta for a specific observation i
                theta1[j,0] = theta1[j,0] + alpha*(Y1_p[i,0]-H1[i,0]) * X1[i,j] - L*theta1[j,0]
                theta2[j,0] = theta2[j,0] + alpha*(Y2_p[i,0]-H2[i,0]) * X2[i,j] - L*theta2[j,0]
                theta3[j,0] = theta3[j,0] + alpha*(Y3_p[i,0]-H3[i,0]) * X3[i,j] - L*theta3[j,0]
    
    theta = numpy.hstack((theta0,theta1,theta2,theta3)) # this is the converged theta matrix for each K fold, each column corresponds to one K fold
    
    
    # This computes the H vectors with the test set which is the K folf that was 
    # left out and adapted to hold the five best features
    H0 = S.sigmoid_vector(X0test,theta0) 
    H1 = S.sigmoid_vector(X1test,theta1)
    H2 = S.sigmoid_vector(X2test,theta2)
    H3 = S.sigmoid_vector(X3test,theta3)
    
    # this calculates the number correct using the outputs associated with the 
    # left out K. It also updates H based on if entry is greater than or less than
    # zero 
    correct0 = S.predicted_output(Y0,H0) 
    correct1 = S.predicted_output(Y1,H1)
    correct2 = S.predicted_output(Y2,H2)
    correct3 = S.predicted_output(Y3,H3)
    
    TestError0 = numpy.sum( numpy.power( Y0 - H0 ,2) ) # this sums the square error
    TestError1 = numpy.sum( numpy.power( Y1 - H1 ,2) )
    TestError2 = numpy.sum( numpy.power( Y2 - H2 ,2) )
    TestError3 = numpy.sum( numpy.power( Y3 - H3 ,2) )
    CVError_total = TestError0 + TestError1 + TestError2 + TestError3
    CVError = numpy.append(CVError,CVError_total)
    
    correct_total = 100*(correct0 + correct1 + correct2 + correct3) / (Y0.shape[0] + Y1.shape[0] + Y2.shape[0]  +Y3.shape[0]  )
    correctness = numpy.append(correctness,correct_total)
    
    lambda_vector = numpy.append(lambda_vector,L)
    L = L + 0.1 # update lambda
    
    
# You want to graph the correctness and the CVError to find the maximum of the 
# correctness and the minimum of the CVError. It turns out the arguement of both
# function is at the same lambda. The ideal Lambda is 0.4.
plt.plot(lambda_vector,correctness, 'r') #, lambda_vector, CVError, 'b')
plt.xlablel =('lambda')
plt.ylablel =('%correctness/CVError')
plt.title('Finding the global lambda value')
plt.legend()
plt.show()

print('Based on the graph the optimal lambda occurs at lambda = 0.4')

            
    
    
    


        
    