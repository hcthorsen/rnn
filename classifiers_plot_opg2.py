print(__doc__)


# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause


import pandas as pd
from pylab import *
from scipy.stats import zscore
from scipy import stats
from scipy.stats import norm


import xlrd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Decision Tree", "Artificial Neural Net", "Naive Bayes", "Logistic regression"]

classifiers = [
    KNeighborsClassifier(15),
    DecisionTreeClassifier(max_depth=3),
    MLPClassifier(alpha=1),
    GaussianNB(),
    LogisticRegression(C=5)]


# load data from saheart
from load_saheart import *

# Make new observation
x_fake = np.matrix([140,4,2,25,0,0,25,14,42])
# Add the new observation to X
X = np.r_[X, x_fake]
N+=1

# Subtract mean value from data
#Y = X - np.ones((N,1))*X.mean(0)
Y = (X - np.ones((N,1))*X.mean(0))/(np.ones((N,1))*X.std(0))

# Remove the new observation from X
x_fake = Y[462,:]
Y = Y[0:462,:]
N-=1
# PCA by computing SVD of Y
U,S,V = linalg.svd(Y,full_matrices=False)
V = mat(V).T

# Project the centered data onto principal component space
Z = Y * V

# Project the observation onto the two first principal components 
z = x_fake*V
z = z[:,0:2]

yy=np.asarray(y).ravel()
ds = (np.asarray(Z[:,0:2]),yy)

figure = plt.figure(figsize=(18, 9))
i = 1
# iterate over datasets
#for ds_cnt, ds in enumerate(datasets):

# preprocess dataset, split into training and test part
X, y = ds
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.2)

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.subplot(2, 3, i)
ax.set_title("Input data")
# Plot the training points
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
# and testing points
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())

i=2
# iterate over classifiers
for name, clf in zip(names, classifiers):
    ax = plt.subplot(2, 3, i)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot also the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
               alpha=0.6)
    # Plot new observation
    ax.scatter(z[:, 0], z[:, 1], c='yellow', s=80, cmap=cm_bright, marker='^')
    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(name)
    #ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            #size=15, horizontalalignment='right')
    i += 1

plt.tight_layout()
plt.show()