#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 20:53:50 2021

@author: daleepsingh
"""
import pickle 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
with open('mnistTVT.pickle','rb') as f:
          Xtrain,Ttrain,Xval,Tval,Xtest,Ttest = pickle.load(f)
          
import bonnerlib2D as bl2d
from sklearn.utils import resample
from sklearn.mixture import GaussianMixture
import numpy.random as r

#p = Xtrain[1].reshape((28,28))
#plt.imshow(p,cmap = 'Greys',interpolation = "Nearest")
#print(p)
#---------------------------------------------------------------
                        #q1a
#---------------------------------------------------------------
Xsmall = Xtrain[:500]
Tsmall = Ttrain[:500]

#---------------------------------------------------------------
                        #q1b
#---------------------------------------------------------------

np.random.seed(7)
mlp = MLPClassifier(solver = 'sgd',activation = 'logistic',\
                           batch_size=200,learning_rate_init =0.1,\
                               max_iter=10000,shuffle = True,
                               tol = 1e-6, hidden_layer_sizes = (5),\
                                   alpha = 0)

mlp.fit(Xsmall,Tsmall)
print("\n---------Q1b---------: \n")
print("training acc ={0}".format(mlp.score(Xsmall,Tsmall)))
print("validation acc ={0}\n".format(mlp.score(Xval,Tval)))


#---------------------------------------------------------------
                        #q1d
#---------------------------------------------------------------

p = mlp.predict_proba(Xval)
pred = np.argmax(p,axis=1)
accuracy = (np.count_nonzero((pred - Tval) == 0))/10000

print("accuracy of proba_predict - mlp.score = {0}".format(accuracy - mlp.score(Xval,Tval)))
#Q1e)
#when given multiple input arrays, resample will produce the same resampling
#method for both arrays. Thus, if we pass in the training and test set into
#resample, the targets will match up with the correct training inputs.
l = []
accuracy_l = []
np.random.seed(7)
for i in range(100):
    r = resample(Xtrain,Ttrain,n_samples=500)
    mlp.fit(r[0],r[1])
    p = mlp.predict_proba(Xval)
    l.append(p)
    avg = np.mean(l,axis=0)
    pred = np.argmax(avg,axis=1)
    accuracy_l.append((np.count_nonzero((pred - Tval) == 0))/10000)

plt.plot(accuracy_l)
plt.xlabel("iteration")
plt.ylabel("accuracy")
plt.show()

#---------------------------------------------------------------
                        #q2a
#---------------------------------------------------------------

#make 10x10 array of 0's (represents background, navy blue)
env = (np.zeros((10,10)))

#insert starting point represented as 1 (blue green)
env[5,3] = 1

#insert ending point represented as 2 (bright green)
env[5,6] = 2

#insert barriers represented as 3 (orange)
env[7,2:6] = 3
env[2:8,5] = 3
env[3,1:3] = 3
env[4,5:9] = 3
env[7,8]   = 3

plt.imshow(env)
plt.title("Question 2(a): Grid World")

#---------------------------------------------------------------
                        #q2b
#---------------------------------------------------------------

#define edges: left,top,right,bot with values -1,-2,-3,-4, respectively
env[:,0]  = -1 #left
env[0,:]  = -2 #top
env[:,-1] = -3 #right
env[-1,:] = -4 #bot

#corners
env[0,0] = -5 #top left corner
env[0,-1] = -6 #top right corner
env[-1,0] = -7 #bot left corner
env[-1,-1] = -8 #bot right corner

#obtain indicies of background,start,end,barrier,edges
bkgrnd = np.argwhere(env == 0)
startpt = np.argwhere(env == 1)
endpt = np.argwhere(env == 2)
barrier = np.argwhere(env == 3)
left_edge = np.argwhere(env == -1)
top_edge = np.argwhere(env ==-2)
right_edge = np.argwhere(env == -3)
bot_edge = np.argwhere(env == -4)
topleft_corner = np.argwhere(env == -5)
topright_corner = np.argwhere(env == -6)
botleft_corner = np.argwhere(env == -7)
botright_corner = np.argwhere(env == -8)

corners = botright_corner,botleft_corner,topleft_corner,topright_corner

#global variables
reward = 0
def Trans(L,a):
    ''' L: Location of agent, action. Only 4 actions: left,right,up,down.'''
    global reward
    
    #edge cases
    if (L == left_edge) and (a == "left"):
        return(L,-1)

    if (L == top_edge) and (a == "top"):
        return(L,-1)
        
    if (L == right_edge) and (a == "right"):
        return(L,-1)

    if (L == bot_edge) and (a == "bot"):
        return(L,-1)

    #corners
    if (L == topleft_corner) and (a == "left" or a == "top"):
        return(L,-1)

    if (L == topright_corner) and (a == "top" or a == "right"):
        return(L,-1)
    
    if (L == botleft_corner) and (a == "left" or a == "down"):
        return(L,-1)
        
    if (L == botright_corner) and (a == "right" or a == "down"):
        return(L,-1)
    
    #barriers
    #recall that L should have shape (2,). L[0] = row, L[1] = col
    #ie L is in Pos [3,2]
    if (L == bkgrnd or L == startpt) and (a == "top"):
        if L in barrier:
            return(L,-1)
        
    if (L == bkgrnd or L == startpt) and (a == "left"):
        if L in barrier:
            return(L,-1)
    
    if (L == bkgrnd or L == startpt) and (a == "right"):
        if L in barrier:
            return(L,-1)
    
    if (L == bkgrnd or L == startpt) and (a == "bot"):
        if L in barrier:
            return(L,-1)
    
    #actual movements
    if (L == bkgrnd or L == startpt) and (a == "top"):
        L[1] +=1
        if L in endpt:
            L = endpt
            return(L, 25)
        else: return (L,0)
        
    if (L == bkgrnd or L == startpt) and (a == "right"):
        if L in endpt:
            L = endpt
            return(L,25)
        else: return (L,0)
        
        reward = 0
    if (L == bkgrnd or L == startpt) and (a == "left"):
        if L in endpt:
            L = endpt
            return(L, 25)
        else: return (L,0)
        
        reward = 0
    if (L == bkgrnd or L == startpt) and (a == "bot"):
        if L in endpt:
            L = endpt
            return(L, 25)
        else: return (L,0)

def choose(L,beta):
#    Qtable = np.zeros((10,10,4)    
    return None

#For the rest of q2 i invoke i dont know policy
#---------------------------------------------------------------
                        #q3a
#---------------------------------------------------------------
'''
-only required to use Xtrain and Xtest.
-Responsibility is the assignment of a data point to a cluster. Thus it is
labels_ in the sklearn module.
'''

with open('cluster_data.pickle','rb') as file:
    dataTrain,dataTest = pickle.load(file)
    Xtrain,Ttrain = dataTrain
    Xtest,Ttest = dataTest


def one_hot(Tint):
    Tint = np.array((Tint))
    max = Tint.max()+1 # num of classes in Tint
    C = np.arange(max)
    C = C.reshape((1,C.shape[0])) #turn C into [1,J]
    Tint = Tint.reshape((Tint.shape[0],1)) #turn Tint into [N,1]
    Thot = Tint == C
    Thot = Thot*1
    return Thot
#print(one_hot(np.array(([4,3,2,3,2,1,2,1,0]))))

clust = KMeans(n_clusters = 3).fit(Xtrain)
#one_hot(cluster.labels_) required to get it into N*3 form (for bl2d).
#Labels_ is the responsibility as it is the assignment of cluster to data point.
#one_hot(cluster.labels_) is the true r_k^(n) assignment as from lecture.
bl2d.plot_clusters(Xtrain, one_hot(clust.labels_)) 
plt.scatter(clust.cluster_centers_[0:,0],clust.cluster_centers_[0:,1],color = "black")
plt.title("Question 3(a): K means")

print("\n---------Q3a---------: \n")
print("score of Xtrain = {0}".format(clust.score(Xtrain)))
print("score of Xtest = {0}".format(clust.score(Xtest)))
#---------------------------------------------------------------
                        #q3b
#---------------------------------------------------------------

gm = GaussianMixture(n_components=3, random_state=0,covariance_type= 'diag',\
                     tol=10**(-7)).fit(Xtrain)
    
rb = gm.predict(Xtrain)
#proba done here instead of one_hot in order to make the color gradient.
#note color gradient is handled in bl2d not in gm.
proba = gm.predict_proba(Xtrain) 
bl2d.plot_clusters(Xtrain,proba) 
plt.scatter(gm.means_[0:,0],gm.means_[0:,1],color = "black")
plt.title("Question 3(b): Gaussian mixture model (diagonal)")

print("\n---------Q3b---------: \n")
print("score of Xtrain ={0}".format(gm.score(Xtrain)))
print("score of Xtest ={0}".format(gm.score(Xtest)))

#---------------------------------------------------------------
                        #q3c
#---------------------------------------------------------------

gm1 = GaussianMixture(n_components=3, random_state=0,covariance_type= 'full',\
                     tol=10**(-7)).fit(Xtrain)
    
rb1 = gm1.predict(Xtrain)
#proba done here instead of one_hot in order to make the color gradient.
#note color gradient is handled in bl2d not in gm.
proba1 = gm1.predict_proba(Xtrain) 
bl2d.plot_clusters(Xtrain,proba1) 
plt.scatter(gm1.means_[0:,0],gm1.means_[0:,1],color = "black")
plt.title("Question 3(c): Gaussian mixture model (full)")
print("\n---------Q3c---------: \n")
print("score of Xtrain ={0}".format(gm1.score(Xtrain)))
print("score of Xtest ={0}".format(gm1.score(Xtest)))
print("Q3c-Q3b test scores = {0}".format((gm1.score(Xtest)-gm.score(Xtest))))


#---------------------------------------------------------------
                        #q3e
#---------------------------------------------------------------
def gmm(X,K,I):
    X = X.T #2*2000
    #Initialization
    #Init mean matrix
    m = r.randint(-10,13,size = (K,2)) #r = np.random, cause np.random alone doesnt work...
    
    #init pi (uniform probability)
    pi = np.ones((K,1))
    pi = pi*(1/K)
    
    #init covariance
    sigma = np.cov(Xtrain) #used simply to find reasonable init value
    
    #K covariance matricies of size 2*2
    sigma = r.randint(sigma.min(),sigma.max(),(K,X.shape[0],X.shape[0]))
  
    Pr_matrix = np.ones((X.shape[0],K))
    t = P(X[0:,0],m[0],sigma[0]) #sending in |X| = (2*1)

    #EM algorithm
    for i in range(I):
        return None

def P(x,m,sigma):
    '''Takes x,mean,sigma, returns p(x) according to gaussian'''
    x = np.reshape(x,(2,1))
    m = np.reshape(m,(2,1))
    M = x.shape[0]
    sigma_inv = 1/(sigma) 
    X_T_SigmaInv = np.matmul(np.dot(-1,(x-m).T),sigma_inv)
    p = (np.exp(np.matmul(X_T_SigmaInv,(x-m)/2)))/((((2*np.pi)**M)*(np.prod(sigma)))**(1/2))
    return p

def r_k_n(k,n,x,m,sigma):
    numerator = (np.pi[k])*P(x,m,sigma)
    denominator = np.sum() #think i need a for loop here...
#could make matrix of p(x)'s, multiply them by corrosponding pi row,
#then sum them.

#---------------------------------------------------------------
                        #Q3h
#---------------------------------------------------------------

with open('mnistTVT.pickle','rb') as f:
          Xtrain,Ttrain,Xval,Tval,Xtest,Ttest = pickle.load(f)
          
Xsmall = Xtrain[:500]
Tsmall = Ttrain[:500]

gm = GaussianMixture(n_components=10, random_state=0,covariance_type= 'diag',\
                     tol=10**(-3)).fit(Xtrain)
    

plt.subplot(4,3,1)
p = np.reshape(gm.means_[0],(28,28))
plt.imshow(p)
for i in range(2,10):
    plt.subplot(4,3,i)
    p = np.reshape(gm.means_[i-1],(28,28))
    plt.imshow(p)

plt.subplot(4,3,10)
p = np.reshape(gm.means_[9],(28,28))
plt.imshow(p)

print("\n---------Q3h---------: \n")
print("score of Xtrain ={0}".format(gm.score(Xtrain)))
print("score of Xtest ={0}".format(gm.score(Xtest)))

plt.suptitle("Question 3(h): mean vectors for 50,000 MNIST training points")

#---------------------------------------------------------------
                        #Q3i
#---------------------------------------------------------------
plt.clf()

gm = GaussianMixture(n_components=10, random_state=0,covariance_type= 'diag',\
                     tol=10**(-3)).fit(Xsmall)
    

plt.subplot(4,3,1)
p = np.reshape(gm.means_[0],(28,28))
plt.imshow(p)
for i in range(2,10):
    plt.subplot(4,3,i)
    p = np.reshape(gm.means_[i-1],(28,28))
    plt.imshow(p)

plt.subplot(4,3,10)
p = np.reshape(gm.means_[9],(28,28))
plt.imshow(p)

print("\n---------Q3i---------: \n")
print("score of Xtrain ={0}".format(gm.score(Xsmall)))
print("score of Xtest ={0}".format(gm.score(Xtest)))

plt.suptitle("Question 3(i): mean vectors for 500 MNIST training points")

#---------------------------------------------------------------
                        #Q3j
#---------------------------------------------------------------
plt.clf()
gm = GaussianMixture(n_components=10, random_state=0,covariance_type= 'diag',\
                     tol=10**(-3)).fit(Xsmall[:10])
    

plt.subplot(4,3,1)
p = np.reshape(gm.means_[0],(28,28))
plt.imshow(p)
for i in range(2,10):
    plt.subplot(4,3,i)
    p = np.reshape(gm.means_[i-1],(28,28))
    plt.imshow(p)

plt.subplot(4,3,10)
p = np.reshape(gm.means_[9],(28,28))
plt.imshow(p)

print("\n---------Q3j---------: \n")
print("score of Xtrain ={0}".format(gm.score(Xsmall[:10])))
print("score of Xtest ={0}".format(gm.score(Xtest)))

plt.suptitle("Question 3(j): mean vectors for 10 MNIST training points")
