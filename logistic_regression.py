"""EECS545 HW2: Logistic Regression."""

import numpy as np
import math


def hello():
    print('Hello from logistic_regression.py')


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def naive_logistic_regression(X: np.ndarray, Y: np.ndarray, max_iters = 100) -> np.ndarray:
    """Computes the coefficients w from the datset (X, Y).

    This implementation uses a naive set of nested loops over the data.
    Specifically, we are required to use Newton's method (w = w - inv(H)*grad).

    Inputs:
      - X: Numpy array of shape (num_data, num_features+1).
           The first column of each row is always 1.
      - Y: Numpy array of shape (num_data) that has 0/1.
      - max_iters: Maximum number of iterations
    Returns:
      - w: Numpy array of shape (num_features+1) w[i] is the coefficient for the i-th
           column of X. The dimension should be matched with the second dimension of X.
    """
    N, d = X.shape
    w = np.zeros(d, dtype=X.dtype)
    for iter in range(max_iters):
        grad = np.zeros(d)
        H = np.zeros((d, d))
        for data_x, data_y in zip(X, Y):
            ###################################################################
            # TODO: Implement this function using Newton's method.            #
            # Specifically, you are required to update the following two      #
            # variables described below.                                      #
            # * grad: accumulated gradient over data samples.                 #
            # * H: accumulated Hessian matrix over data samples.              #
            # Please do not check a convergence condition for this time.      #
            # note: You are allowed to use predefined sigmoid function above. #
            ###################################################################
            #raise NotImplementedError("TODO: Add your implementation here.")
            h_xi = sigmoid(np.dot(data_x, w))
            grad = grad + data_x*(data_y-h_xi)
            H = H - h_xi*(1-h_xi)*np.matmul(np.transpose(data_x.reshape((1,d))), data_x.reshape((1,d)))
            ###################################################################
            #                        END OF YOUR CODE                         #
            ###################################################################
        w = w - np.matmul(np.linalg.inv(H), grad)
    return w


def vectorized_logistic_regression(X: np.ndarray, Y: np.ndarray, max_iters = 100) -> np.ndarray:
    """Computes the coefficients w from the dataset (X, Y).

    This implementation will vectorize the implementation in naive_logistic_regression,
    which implements Newton's method (w = w - inv(H)*grad).

    Inputs:
      - X: Numpy array of shape (num_data, num_features+1).
           The first column of each row is always 1.
      - Y: Numpy array of shape (num_data) that has 0/1.
      - max_iters: Maximum number of iterations
    Returns:
      - w: Numpy array of shape (num_features+1) w[i] is the coefficient for the i-th
           column of X. The dimension should be matched with the second dimension of X.
  """
    N, d = X.shape
    w = np.zeros(d, dtype=X.dtype)
    for iter in range(max_iters):
        #######################################################################
        # TODO: Implement this function using Newton's method.                #
        # Because this version does not have any inner loop, compared to the  #
        # previous (naive_logistic_regression) task, you are required to      #
        # compute grad and H at once.                                         #
        # * grad: gradient from all data samples.                             #
        # * H: Hessian matrix from all data samples.                          #
        # The shape of grad and H is the same as 'naive_logistic_regression'. #
        # Please do not check a convergence condition for this time.          #
        # You are allowed to use predefined sigmoid function above.           #
        #######################################################################
        grad = X.T@(Y - sigmoid(X@w))  # hint: grad.shape should be (d, ) at the end of this block
        H = - X.T@ (np.expand_dims((sigmoid(X@w) * (1 - sigmoid(X@w))), -1) * X)  # hint: H.shape should be (d, d) at the end of this block
        #raise NotImplementedError("TODO: Add your implementation here.")
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        w = w - np.matmul(np.linalg.inv(H), grad)
    return w


def compute_y_boundary(X_coord: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Computes the matched y coordinate value for the decision boundary from
    the x coordinate and coefficients w.

    Inputs:
      - X_coord: Numpy array of shape (d, ). List of x coordinate values.
      - w: Numpy array of shape (3, ) that stores the coefficients.

    Returns:
      - Y_coord: Numpy array of shape (d, ).
                 List of y coordinate values with respect to the coefficients w.
    """
    Y_coord = None
    ###########################################################################
    # TODO: Compute y_coordinate of the decision boundary with respect to     #
    # x_coord and coefficients w. Please return/save your y_coordindate into  #
    # y_coord parameter. It is fair to assume that w[2] will not be zero.     #
    ###########################################################################
    #raise NotImplementedError("TODO: Add your implementation here.")
    Y_coord = - w[0]/w[2] - w[1]/w[2]*X_coord
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

    return Y_coord
