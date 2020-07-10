"""Functions for iterative methods notebook."""
from time import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def gradient_descent(func, grad, x0, step, tol=1e-6, max_iter=1000):
    """Minimize function with gradient descent.

    Parameters
    ----------
    func : function
        Objective function to minimize.
    grad : function
        Gradient of objective function.
    x0 : array
        Starting point for solver.
    step : float
        Step size for gradient step.
    tol : float, optional
        Gradient tolerance for terminating solver.
    max_iter : int, optional
        Maximum number of iterations for solver.

    Returns
    -------
    x : float
        Function minimizer.
    x_vals : array
        Iterates.
    func_vals : array
        Function values at iterates.
    grad_vals : array
        Norm of gradient at iterates.

    """
    # Initialize return values
    if type(x0) in (int, float):
        x_vals = np.zeros((1, max_iter + 1))
    else:
        x_vals = np.zeros((len(x0), max_iter + 1))
    x_vals[:, 0] = x0
    func_vals = np.zeros(max_iter + 1)
    func_vals[0] = func(x0)
    grad_vals = np.zeros(max_iter + 1)
    grad_vals[0] = np.linalg.norm(grad(x0))

    # Minimize function
    start = time()
    for ii in range(1, max_iter + 1):
        x_vals[:, ii] = x_vals[:, ii - 1] - step*grad(x_vals[:, ii - 1])
        func_vals[ii] = func(x_vals[:, ii])
        grad_vals[ii] = np.linalg.norm(grad(x_vals[:, ii]))

        # Check convergence
        if grad_vals[ii] < tol:
            print(f'Converged after {ii} iteration(s).')
            print(f'Minimum function value: {min(func_vals[:(ii + 1)]):.2f}')
            print(f'Total time: {time() - start:.2f} secs')
            return x_vals[:, ii], x_vals[:, :(ii + 1)].squeeze(), \
                   func_vals[:(ii + 1)], grad_vals[:(ii + 1)]

    print(f'Maximum number of iterations reached: {ii}.')
    print(f'Minimum function value: {min(func_vals[:(ii + 1)]):.2f}')
    print(f'Total time: {time() - start:.2f} secs')
    return x_vals[:, -1], x_vals.squeeze(), func_vals, grad_vals


def stochastic_descent(A, y, learning_rate, decay=None, batch_size=1,
                       tol=1e-6, max_iter=1000, random_seed=1):
    """Solve linear least-squares with stochastic gradient descent.

    Parameters
    ----------
    A : numpy.ndarray
        Training data coefficient matrix.
    y : numpy.ndarray
        Training data solution vector.
    learning_rate : float
        Initial learning rate for stochastic gradient descent.
    decay : float, optional
        Learning rate schedule decay parameter.
    batch_size : int, optional
        Number of training examples used to approximate gradient.
    tol : float, optional
        Difference between iterates tolerance for terminating solver.
    max_iter : int, optional
        Maximum number of iterations for solver.
    random_seed : int, optional
        Random number generator seed for reproducibility.

    Returns
    -------
    x : float
        Function minimizer.
    x_vals : array
        Iterates.
    func_vals : array
        Function values at iterates.
    func_diff : array
        Difference between subsequent iterates.

    """
    m, n = A.shape
    if batch_size < 1:
        print(f'Invalid batch_size {batch_size}, using 1.')
        batch_size = 1
    if batch_size > m:
        print(f'Invalid batch_size {batch_size}, using {m}.')
        batch_size = m

    # Initialize return values
    np.random.seed(random_seed)
    x_vals = np.zeros((n, max_iter + 1))
    x_vals[:, 0] = np.random.randn(n)
    func_vals = np.zeros(max_iter + 1)
    func_vals[0] = np.linalg.norm(A.dot(x_vals[:, 0]) - y)**2/m
    func_diff = np.zeros(max_iter)

    # Minimize function
    start = time()
    for ii in range(1, max_iter + 1):
        idx = np.random.randint(0, m, batch_size)
        res = A[idx, :].dot(x_vals[:, ii - 1]) - y[idx]
        grad = 2/batch_size*np.transpose(A[idx, :]).dot(res)
        x_vals[:, ii] = x_vals[:, ii - 1] - learning_rate*grad
        func_vals[ii] = np.linalg.norm(A.dot(x_vals[:, ii]) - y)**2/m
        func_diff[ii - 1] = np.abs(func_vals[ii] - func_vals[ii - 1])

        # Check convergence
        if func_diff[ii - 1] < tol:
            print(f'Converged after {ii} iteration(s).')
            print(f'Minimum function value: {min(func_vals[:(ii + 1)]):.2f}')
            print(f'Total time: {time() - start:.2f} secs')
            return x_vals[:, ii], x_vals[:, :(ii + 1)].squeeze(), \
                   func_vals[:(ii + 1)], func_diff[:(ii + 1)]

        # Update learning rate
        if decay is not None:
            learning_rate = learning_rate/(1 + decay*ii)

    print(f'Maximum number of iterations reached: {ii}.')
    print(f'Minimum function value: {min(func_vals):.2f}')
    print(f'Total time: {time() - start:.2f} secs')
    return x_vals[:, -1], x_vals.squeeze(), func_vals, func_diff


def plot_1d(func, results):
    """Plot 1D results from gradient_descent().

    Parameters
    ----------
    func : function
        Function to be minimized.
    results : list
        Results from gradient_descent().

    Returns
    -------
    None.

    """
    # Plot set up
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    cmap = 'viridis_r'
    norm = plt.Normalize(0, len(results[1]) - 1)
    pad = (max(results[1]) - min(results[1]))/10
    x_vals = np.linspace(min(results[1]) - pad, max(results[1]) + pad)

    # Plot iterates
    ax[0].plot(x_vals, func(x_vals))
    ax[0].scatter(results[1], results[2], c=np.arange(0, len(results[1])),
                  zorder=3, cmap=cmap, norm=norm)
    ax[0].set_xlabel('$x$')
    ax[0].set_ylabel('$f(x)$')
    fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), label='Iteration',
                 ax=ax[0])

    # Plot function values
    ax[1].plot(results[2])
    ax[1].scatter(np.arange(len(results[2])), results[2], zorder=3, cmap=cmap,
                  c=np.arange(0, len(results[1])), norm=norm)
    ax[1].set_xlabel('Iteration ($k$)')
    ax[1].set_ylabel('$f(x^k)$')

    # Plot norm of gradient values
    ax[2].plot(results[3])
    ax[2].scatter(np.arange(len(results[3])), results[3], zorder=3, cmap=cmap,
                  c=np.arange(0, len(results[1])), norm=norm)
    ax[2].set_xlabel('Iteration ($k$)')
    ax[2].set_ylabel(r'$||\nabla f(x^k)||$')
    plt.show()


def plot_2d(func, results):
    """Plot 2D results from gradient_descent().

    Parameters
    ----------
    func : function
        Function to be minimized.
    results : list
        Results from gradient_descent().

    Returns
    -------
    None.

    """
    # Plot set up
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    x_vals = np.linspace(*_get_bounds(results))
    X = np.meshgrid(x_vals, x_vals)
    norm = plt.Normalize(np.min(func(X)), np.max(func(X)))

    # Plot iterates
    ax[0].contour(x_vals, x_vals, func(X))
    ax[0].plot(results[1][0, :], results[1][1, :], '.')
    ax[0].set_xlabel('$x$')
    ax[0].set_ylabel('$y$')
    fig.colorbar(cm.ScalarMappable(norm=norm), label='$f(x, y)$', ax=ax[0])

    # Plot function values
    ax[1].plot(results[2])
    ax[1].set_xlabel('Iteration ($k$)')
    ax[1].set_ylabel('$f(x^k, y^k)$')

    # Plot norm of gradient values
    ax[2].plot(results[3])
    ax[2].set_xlabel('Iteration ($k$)')
    ax[2].set_ylabel(r'$||\nabla f(x^k, y^k)||$')


def _get_bounds(results):
    """Get upper and lower bounds for 2D plot."""
    max_val = max(max(results[1][0]), max(results[1][1]))
    min_val = min(min(results[1][0]), min(results[1][1]))
    bound = max(abs(max_val), abs(min_val))
    pad = bound/5
    return [-bound - pad, bound + pad]


def plot_sgd(results):
    """Plot results from stochastic_descent().

    Parameters
    ----------
    results : list
        Results from stochastic_descent().

    Returns
    -------
    None.

    """
    # Plot function values
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(results[2])
    ax[0].set_xlabel('Iteration ($k$)')
    ax[0].set_ylabel('$f(x^k)$')

    # Plot difference between function values
    ax[1].plot(np.arange(1, len(results[3]) + 1), results[3])
    ax[1].set_xlabel('Iteration ($k$)')
    ax[1].set_ylabel('$|f(x^k) - f(x^{k-1})|$')
