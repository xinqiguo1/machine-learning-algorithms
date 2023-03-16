"""EECS545 HW1: Linear Regression."""

from typing import Any, Dict, Tuple

import numpy as np


def load_data():
    """Load the data required for Q2."""
    x_train = np.load('data/q2xTrain.npy')
    y_train = np.load('data/q2yTrain.npy')
    x_test = np.load('data/q2xTest.npy')
    y_test = np.load('data/q2yTest.npy')
    return x_train, y_train, x_test, y_test


def generate_polynomial_features(x: np.ndarray, M: int) -> np.ndarray:
    """Generate the polynomial features.

    Args:
        x: A numpy array with shape (N, ).
        M: the degree of the polynomial.
    Returns:
        phi: A feature vector represented by a numpy array with shape (N, M+1);
          each row being (x^{(i)})^j, for 0 <= j <= M.
    """
    N = len(x)
    phi = np.zeros((N, M + 1))
    for m in range(M + 1):
        phi[:, m] = np.power(x, m)
    return phi


def loss(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    r"""The least squares training objective for the linear regression.

    Args:
        X: the feature matrix, with shape (N, M+1).
        y: the target label for regression, with shape (N, ).
        w: the linear regression coefficient, with shape (M+1, ).
    Returns:
        The least square error term with respect to the coefficient weight w,
        E(\mathbf{w}).
    """
    y_pred = np.matmul(X, w)
    squared_error = (y - y_pred) ** 2
    return np.sum(squared_error)


def MSE(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Returns mean squared error for the linear regression.

    Args:
        X: the feature matrix, with shape (N, M+1).
        y: the target label for regression, with shape (N, ).
        w: the linear regression coefficient, with shape (M+1, ).
    Returns:
        The mean squared error with respect to the coefficient weight w.
    """
    y_pred = np.matmul(X, w)
    squared_error = (y - y_pred) ** 2
    mse = np.mean(squared_error)
    return mse


def batch_gradient_descent(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    eta: float = 0.01,
    max_epochs: int = 10000,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Batch gradient descent for linear regression that fits the
    feature matrix `X_train` to target `y_train`.

    Args:
        X_train: the feature matrix, with shape (N, M+1).
        y_train: the target label for regression, with shape (N, ).
        eta: Learning rate.
        max_epochs: Maximum iterations (epochs) allowed.
    Returns: A tuple (w, info)
        w: The coefficient of linear regression found by GD. Shape (M+1, ).
        info: A dict that contains additional information (see the notebook).
    """
    ###################################################################
    # TODO: Implement the Batch GD solver.
    ###################################################################
    raise NotImplementedError("TODO: Add your implementation here.")
    ###################################################################
    #                        END OF YOUR CODE                         #
    ###################################################################
    return w, info


def stochastic_gradient_descent(
    X_train: np.ndarray,
    y_train: np.ndarray,
    eta=4e-2,
    max_epochs=10000,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Stochastic gradient descent for linear regression that fits the
    feature matrix `X_train` to target `y_train`.

    Args:
        X_train: the feature matrix, with shape (N, M+1).
        y_train: the target label for regression, with shape (N, ).
        eta: Learning rate.
        max_epochs: Maximum iterations (epochs) allowed.
    Returns: A tuple (w, info)
        w: The coefficient of linear regression found by SGD. Shape (M+1, ).
        info: A dict that contains additional information (see the notebook).
    """
    ###################################################################
    # TODO: Implement the SGD solver.
    ###################################################################
    raise NotImplementedError("TODO: Add your implementation here.")
    ###################################################################
    #                        END OF YOUR CODE                         #
    ###################################################################
    return w, info


def closed_form(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    lam: float = 0.0,
) -> np.ndarray:
    """Return the closed form solution of linear regression.

    Arguments:
        X_train: The X feature matrix, shape (N, M+1).
        y_train: The y vector, shape (N).
        M: The degree of the polynomial to generate features for.
        lam: The regularization coefficient lambda.

    Returns:
        The (optimal) coefficient w for the linear regression problem found,
        a numpy array of shape (M+1, ).
    """
    ###################################################################
    # TODO: Implement the closed form solution.
    ###################################################################
    raise NotImplementedError("TODO: Add your implementation here.")
    ###################################################################
    #                        END OF YOUR CODE                         #
    ###################################################################
    return w


def closed_form_locally_weighted(
    X_train: np.ndarray,
    y_train: np.ndarray,
    r_train: np.ndarray,
) -> np.ndarray:
    """Return the closed form solution of locally weighted linear regression.

    Arguments:
        x_train: The X feature matrix, shape (N, M+1).
        y_train: The y vector, shape (N, ).
        r_train: The local weights for data point. Shape (N, ).

    Returns:
        The (optimal) coefficient for the locally weighted linear regression
        problem found. A numpy array of shape (M+1, ).
    """

    ###################################################################
    # TODO: Implement the closed form solution.
    ###################################################################
    raise NotImplementedError("TODO: Add your implementation here.")
    ###################################################################
    #                        END OF YOUR CODE                         #
    ###################################################################
    return w
