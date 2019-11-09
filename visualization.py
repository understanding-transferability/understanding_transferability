import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(predictor, xs, ys):
    '''
    plot 2D decision boundary for a classifier.
    ``predictor`` shoule be a function, which can be called with ``predictor(data)``, where data has (n_sample, 2) shape
    ``xs`` should be a sequence representing the grid's x coordinates
    ``ys`` should be a sequence representing the grid's y coordinates

    example usage::

    plot_decision_boundary(func, np.arange(-6.0, 8.0, 0.1), np.arange(-8.0, 8.0, 0.1))

    '''
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    data = np.asarray([xx.flatten(), yy.flatten()]).T
    label = predictor(data)
    label.resize(xx.shape)
    plt.contourf(xx, yy, label, alpha=0.3)