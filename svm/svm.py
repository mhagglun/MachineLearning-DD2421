import random, math
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")


class SVM(object):
    def __init__(self): #TODO: Select kernel
        pass
    
    # Fit model to input data X, t is the targets for each point
    def fit(self, X, t):
        self.X = X
        self.t = t
        self.P = Pmatrix(X,t)

        N = X.shape[0]
        self.N = N
        start = np.zeros((N, 1)) # Initial guess
        C = 300 # Can assign None = no upper bound
        bounds  = [(0, C) for b in range(N)]
        cons = {'type':'eq', 'fun': self.zerofun}

        # Call minimize
        # Find alpha that minimizes problem
        ret = minimize(self.objective, start, bounds=bounds, constraints=cons)
        if (not ret['success']):
            print("Cannot find optimizing solution")
        alpha = ret['x']

        # Extract indicies of non-zero alpha values
        indicies = np.argwhere(np.abs(alpha) >= 10 ** (-5))

        # Save alpha_i with corresponding data points x_i and target values t_i
        self.sol = np.array([ [alpha[idx], X[idx][0], t[idx]] for idx in indicies])
    

        # Get a support vector sv
        sv = self.sol[0] # First support vector

        # print('Alpha value: ' + str(np.squeeze(sv[0])) + " vs C value: "+str(C))

        # all_svs = [sv[1] for sv in self.sol] # Coordinates for all sv
        # sv_x = [sv[0] for sv in all_svs]
        # sv_y = [sv[1] for sv in all_svs]
        # plt.plot(sv_x, sv_y, 'k*')           # Mark the support vectors in the plot

        # Calculate b value
        b = sum([ alpha[i]*t[i]*kernel(sv[1], X[i]) for i in range(N) ]) - sv[2]

        # Set weights (alpha + target class) and bias for model which will allow us to make 
        # predictions with the indicator function

        self.alpha = alpha
        self.b = b

        pass

    # Make predictions on model with new samples X
    def predict(self,X):

        # make calls to indicator function for every data point

        # plot the data points and contour plot

        pass



    # Defines a function which implements (4)
    # @param alpha - vector which minimizes the problem
    def objective(self, alpha):
        return 0.5 * np.dot(alpha, np.dot(alpha.T, self.P)) - sum(alpha)

    # Function which implements the equality constrain of (10)
    # sum(t_i * alpha_i) = 0
    def zerofun(self, alpha):
        return np.dot(self.t, alpha)


    # Indicator function
    def indicator(self,x):       
        return sum([ sv[0] * sv[2] * kernel(sv[1],x) for sv in self.sol ]) - self.b


    def drawPoints(self, X, y):
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', marker='.')
        plt.axis('equal')


    def visualize(self):
        X = self.X
        y = self.t

        xgrid = np.linspace(np.min(X[:,0])*1.5, np.max(X[:,0])*1.5)
        ygrid = np.linspace(np.min(X[:,1])*1.5,np.max(X[:,1])*1.5)

        grid = np.squeeze(np.array([ [self.indicator([x,y])
                            for x in xgrid]
                            for y in ygrid]))

        cdict = {-1: 'red', 1: 'blue'}
        lbldict = {-1: 'Class A', 1:'Class B'}

        # plot all points
        fig, ax = plt.subplots()
        for t in np.unique(y):
            ix = np.where(y == t)[0]
            ax.scatter(X[ix, 0], X[ix, 1], c=cdict[t], label=lbldict[t], s=20, marker='.')
        



        # TODO: Might need some fixing
        # set marker for support vectors
        types = [sv[2] for sv in self.sol]
        svs = np.array([sv[1] for sv in self.sol])
        print(types)
        print(svs)
        for t in np.unique(types):
            ix = np.where(types == t)[0]
            if (t == -1):
                label='Support Vector (A)'
            else:
                label='Support Vector (B)'
            ax.plot(svs[ix,0], svs[ix,1], c=cdict[t], marker='d', markersize=6, linestyle='None', label=label)
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())


        # plot boundaries
        ax.contourf(xgrid, ygrid, grid, (-1, 0, 1), colors=['red','blue'], alpha=0.5)

        # ax.axis('equal')
        plt.show()

# The kernel function computes the scalar value corresponding to phi(s) * phi(xi)
def kernel(x,y):
    # Linear kernel
    return np.dot(x,y)

    # Polynomial kernel
    # p = 4                               # Change this later
    # return (np.dot(x,y) + 1) ** p 

    # RBF kernel
    # sigma = 2
    # return math.exp( (-np.linalg.norm(x-y)**2) / (2*sigma**2) )


# Calculates P matrix
def Pmatrix(X,t):
    N = X.shape[0]
    P = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            P[i,j] = t[i]*t[j]*kernel(X[i,:], X[j,:])
    return P

