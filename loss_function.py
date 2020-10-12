import numpy as np

def _hinge_loss(w, X, y,C):

    
    y_hat = y * np.dot(X,w)
    y_hat = np.maximum(np.zeros_like(y_hat), (1-y_hat)) # hinge function
    
    return C*sum(y_hat)

