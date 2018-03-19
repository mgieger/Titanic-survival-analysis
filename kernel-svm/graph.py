
#modified from src: http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html#sphx-glr-auto-examples-svm-plot-iris-py
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def plot_svm(svm, filename, title, x_axis_title, y_axis_title, data, labels):
    '''
    Saves kernel plot (linear, rbf, ...)
    Args:
        svm (sklearn.svc): svm machine to plot decision boundary
        title (string): title to display on graph / filename
        x_axis_title (string): x axis title
        y_axis_title (string): y axis title
        data (np.array): data of shape (n, 2) where n is number of training samples
        labels (np.array): labels of shape (n, ) for the given data
    Returns:
        None: saves plot of decision boundary to file
    '''
    sub = plt.gca()
    feature_1, feature_2 = data[:, 0], data[:, 1]
    mesh_x, mesh_y = make_meshgrid(feature_1, feature_2)
    plot_contours(sub, svm, mesh_x, mesh_y,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    sub.scatter(feature_1, feature_2, c=labels, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    sub.set_xlim(mesh_x.min(), mesh_x.max())
    sub.set_ylim(mesh_y.min(), mesh_y.max())
    sub.set_xlabel(x_axis_title)
    sub.set_ylabel(y_axis_title)
    sub.set_xticks(())
    sub.set_yticks(())
    sub.set_title(title)
    #save fig and clear
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()
