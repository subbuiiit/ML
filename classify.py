import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

import sklearn
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.externals.six import StringIO
import pydot



### prepare dataset

(X, y) = make_moons(n_samples=100, noise=0.3, random_state=0)
X = StandardScaler().fit_transform(X) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

h=.02 # step size in the mesh
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.subplot(1, 2, 1)
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright) 
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())



## classify using a decision tree 
'''
nonparametric discriminartive learning method. goal is to predict a binary tree based model that
predicts the traget value by learning simple decision tules from the data. Given a training data (X, y), a decision tree recursively partitions the space such that samples with same lables are grouped together.

Controling parameters are max_depth.
loss functions used are gini/entropy to measure impurity of datasplits. 
'''
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
if hasattr(clf, "decision_function"):
   Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
else:
     Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
ax = plt.subplot(1, 2, 2)
# Put the result into a color plot
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

# Plot also the training points
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
# and testing points
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   alpha=0.6)

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
ax.set_title("decision trees")
ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')

plt.show()
