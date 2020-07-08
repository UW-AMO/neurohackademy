"""Functions for gradient descent notebook."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def gradient_descent(x0, func, grad, step, tol=1e-6, max_iter=1000):
    """Minimize function with gradient descent.

    Parameters
    ----------
    x0 : array
        Starting point for solver.
    func : function
        Objective function to minimize.
    grad : function
        Gradient of objective function.
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
    flag : int
        0, norm of gradient below `tol`.
        1, maximum number of iterations reached.

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
    for ii in range(1, max_iter + 1):
        x_vals[:, ii] = x_vals[:, ii - 1] - step*grad(x_vals[:, ii - 1])
        func_vals[ii] = func(x_vals[:, ii])
        grad_vals[ii] = np.linalg.norm(grad(x_vals[:, ii]))

        # Check convergence
        if grad_vals[ii] < tol:
            return x_vals[:, ii], x_vals[:, :(ii + 1)].squeeze(), \
                   func_vals[:(ii + 1)], grad_vals[:(ii + 1)], 0

    return x_vals[:, -1], x_vals.squeeze(), func_vals, grad_vals, 1


def newtons_method(x0, func, grad, hess, tol=1e-6, max_iter=1000):
    """Minimize function with Newton's method.

    Parameters
    ----------
    x0 : array
        Starting point for solver.
    func : function
        Objective function to minimize.
    grad : function
        Gradient of objective function.
    hess : function
        Hessian of objective function.
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
    flag : int
        0, norm of gradient below `tol`.
        1, maximum number of iterations reached.

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
    for ii in range(1, max_iter + 1):
        H = hess(x_vals[:, ii - 1])
        if type(H) in (int, float):
            x_vals[:, ii] = x_vals[:, ii - 1] - grad(x_vals[:, ii - 1])/H
        else:
            x_vals[:, ii] = x_vals[:, ii - 1] - \
                            np.linalg.solve(H, grad(x_vals[:, ii - 1]))
        func_vals[ii] = func(x_vals[:, ii])
        grad_vals[ii] = np.linalg.norm(grad(x_vals[:, ii]))

        # Check convergence
        if grad_vals[ii] < tol:
            return x_vals[:, ii], x_vals[:, :(ii + 1)].squeeze(), \
                   func_vals[:(ii + 1)], grad_vals[:(ii + 1)], 0

    return x_vals[:, -1], x_vals.squeeze(), func_vals, grad_vals, 1


def plot_1d(func, results):
    """Plot 1D results from gradient_descent() or newtons_method().

    Parameters
    ----------
    func : function
        Function to be minimized.
    results : list
        Results from gradient_descent() or newtons_method().

    Returns
    -------
    None.

    """
    # Plot set up
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    cmap = 'viridis_r'
    norm = plt.Normalize(0, len(results[1]) - 1)

    # Plot iterates
    pad = (max(results[1]) - min(results[1]))/10
    x_vals = np.linspace(min(results[1]) - pad, max(results[1]) + pad)
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
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Function Value')

    # Plot norm of gradient values
    ax[2].plot(results[3])
    ax[2].scatter(np.arange(len(results[3])), results[3], zorder=3, cmap=cmap,
                  c=np.arange(0, len(results[1])), norm=norm)
    ax[2].set_xlabel('Iteration')
    ax[2].set_ylabel('Norm of Gradient')
    plt.show()


def plot_2d(func, results):
    """Plot 2D results from gradient_descent() or newtons_method().

    Parameters
    ----------
    func : function
        Function to be minimized.
    results : list
        List of results from gradient_descent() or newtons_method().

    Returns
    -------
    None.

    """
    # Plot set up
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    x_vals = np.linspace(*get_bounds(results, 0))
    y_vals = np.linspace(*get_bounds(results, 1))
    [X, Y] = np.meshgrid(x_vals, y_vals)
    ax[0].contour(x_vals, y_vals, func(X, Y))
    ax[0].set_xlabel('$x$')
    ax[0].set_ylabel('$y$')
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Function Value')
    ax[2].set_xlabel('Iteration')
    ax[2].set_ylabel('Norm of Gradient')
    norm = plt.Normalize(np.min(func(X, Y)), np.max(func(X, Y)))
    fig.colorbar(cm.ScalarMappable(norm=norm), label='$f(x, y)$', ax=ax[0])

    # Plot results
    for ii in range(len(results)):
        ax[0].plot(results[ii][1][0, :], results[ii][1][1, :], '.')
        ax[1].plot(results[ii][2])
        ax[2].plot(results[ii][3])
    plt.show()


def get_bounds(results, ii):
    """Get upper and lower bounds for 2D plot."""
    max_val = max([max(results[jj][1][ii]) for jj in range(len(results))])
    min_val = min([min(results[jj][1][ii]) for jj in range(len(results))])
    bound = max(abs(max_val), abs(min_val))
    pad = bound/5
    return [-bound - pad, bound + pad]
