"""EECS545 HW2: Naive Bayes for Classifying SPAM."""

from typing import Tuple

import numpy as np
import math


def hello():
    print('Hello from naive_bayes_spam.py')


def train_naive_bayes(X: np.ndarray, Y: np.ndarray,
                      ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Computes probabilities for logit x being each class.

    Inputs:
      - X: Numpy array of shape (num_mails, vocab_size) that represents emails.
        The (i, j)th entry of X represents the number of occurrences of the
        j-th token in the i-th document.
      - Y: Numpy array of shape (num_mails). It includes 0 (non-spam) or 1 (spam).
    Returns: A tuple of
      - mu_spam: Numpy array of shape (vocab_size). mu value for SPAM mails.
      - mu_non_spam: Numpy array of shape (vocab_size). mu value for Non-SPAM mails.
      - phi: the ratio of SPAM mail from the dataset email.
    """
    num_mails, vocab_size = X.shape
    spam = X[Y == 1,:]
    non_spam = X[Y == 0,:]
    mu_spam = (spam.sum(axis=0)+1) / (np.sum(spam.sum(axis=1))+vocab_size)
    mu_non_spam = (non_spam.sum(axis=0)+1) / (np.sum(non_spam.sum(axis=1))+vocab_size)
    phi = spam.shape[0] / (spam.shape[0] + non_spam.shape[0])
    ###########################################################################
    # TODO: Compute mu for each word (vocab), for both SPAM and Non-SPAM.     #
    # You also need to compute the phi value. Please check 'Classification 3' #
    # lecture note for how to compute mu and phi.                             #
    # Please do not forget to apply Laplace smothing here.                    #
    ###########################################################################
    #raise NotImplementedError("TODO: Add your implementation here.")
    
    

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return mu_spam, mu_non_spam, phi


def test_naive_bayes(X: np.ndarray,
                     mu_spam: np.ndarray,
                     mu_non_spam: np.ndarray,
                     phi: float,
                     ) -> np.ndarray:
    """Classify whether the emails in the test set is SPAM.

    Inputs:
      - X: Numpy array of shape (num_mails, vocab_size) that represents emails.
        The (i, j)th entry of X represents the number of occurrences of the
        j-th token in the i-th document.
      - mu_spam: Numpy array of shape (vocab_size). mu value for SPAM mails.
      - mu_non_spam: Numpy array of shape (vocab_size). mu value for Non-SPAM mails.
      - phi: the ratio of SPAM mail from the dataset email.
    Returns:
      - pred: Numpy array of shape (num_mails). Mark 1 for the SPAM mails.
    """
    pred = np.zeros(X.shape[0])
    ###########################################################################
    # TODO: Using the mu and phi values, predict whether the mail is SPAM.    #
    # If you implement Naive Bayes in the straightforward way, you will note  #
    # that the computed probability often goes to zero.                       #
    # Hint: Think about using logarithms.                                     #
    ###########################################################################
    #raise NotImplementedError("TODO: Add your implementation here.")
    data_spam = np.matmul(X, np.log(mu_spam))
    data_non_spam = np.matmul(X, np.log(mu_non_spam))
    pred[data_spam + np.log(phi)>data_non_spam + np.log(1-phi)] = 1
         
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return pred


def evaluate(pred: np.ndarray, Y: np.ndarray) -> float:
    """Compute the accuracy of the predicted output w.r.t the given label.

    Inputs:
      - pred: Numpy array of shape (num_mails). It includes 0 (non-spam) or 1 (spam).
      - Y: Numpy array of shape (num_mails). It includes 0 (non-spam) or 1 (spam).
    Returns:
      - accuracy: accuracy value in the range [0, 1].
    """
    accuracy = np.mean((pred == Y).astype(np.float32))

    return accuracy


def get_indicative_tokens(mu_spam: np.ndarray,
                          mu_non_spam: np.ndarray,
                          top_k: int,
                          ) -> np.ndarray:
    """Filter out the most K indicative vocabs from mu.

    We will check the lob probability of mu's. Your goal is to return `top_k`
    number of vocab indices.

    Inputs:
      - mu_spam: Numpy array of shape (vocab_size). The mu value for
                 SPAM mails.
      - mu_non_spam: Numpy array of shape (vocab_size). The mu value for
                     Non-SPAM mails.
      - top_k: The number of indicative tokens to generate. A positive integer.
    Returns:
      - idx_list: Numpy array of shape (top_k), of type int (or int32).
                  Each index represent the vocab in vocabulary file.
    """
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    idx_list = np.zeros(top_k, dtype=np.int32)
    ###################################################################
    # TODO: Get the `top_k` most indicative vocabs.
    ###################################################################
    #raise NotImplementedError("TODO: Add your implementation here.")
    tokens = np.argsort(mu_spam/mu_non_spam)[-top_k:]
    for i in range(top_k):
        idx_list[i] = tokens[i]
    
    ###################################################################
    #                        END OF YOUR CODE                         #
    ###################################################################
    return idx_list

