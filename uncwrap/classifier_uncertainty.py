"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

from uncwrap.models.dirichlet_wrapper import DirichletUncertaintyWrapper


class UncertaintyWrapperEstimator(ClassifierMixin, BaseEstimator):
    """ An Sklearn wrapper that enriches the prediction of a given classifier with the corresponding prediction ucnertainty

    For more information look
    at the paper:
    Mena, J., Pujol, O., & Vitrià, J. (2020).
    Uncertainty-based Rejection Wrappers for Black-box Classifiers. IEEE Access.

    Parameters
    ----------
    black_box : BaseEstimator, default=None
        An object that observes the sklearn BaseEstimator interface. This will be used as the black box to retrieve the original predictions. Optional, if not set, then a list of predictions on X must be passed on fit and predict |
    lambda_reg : float, default=1e-2
        Regularization property used for adjusting the uncertainties.
    num_samples : int, default=1000
        Number of samples to include for the MC sampling of the dirichlet dsitribution.
    learning_rate : float, default='demo'
        Learning rate used when training uncertainties.
    num_hidden_units : int, default='demo'
        Number of units to use in the uncertainty neural network model.
    batch_size : int, default='demo'
        Batch size used when training uncertainties.
    epochs : int, default='demo'
        Number of epochs when training the uncertainties.
    verbose : int, default='demo'
        Verbosity level ofthe training(same as in Keras).

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    wrapper_ : object
        The model used for learning and estimating the uncertainties
    """

    def __init__(self, black_box: BaseEstimator = None, lambda_reg=1e-2, num_samples=1000,
                 learning_rate=1e-3, num_hidden_units=20, batch_size=256, epochs=50, verbose=0):
        self.black_box = black_box
        self.lambda_reg = lambda_reg
        self.num_samples = num_samples
        self.learning_rate = learning_rate
        self.num_hidden_units = num_hidden_units
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose

    def fit(self, X, y, pred_y = None):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        pred_y : array-like, shape (n_samples,)
            Array of predictions corresponding to calls to the blackbox for X.
            It can be used when no access to the representation of the input for the blackbox.
            An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y, accept_sparse='csr')
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        if not self.black_box and not type(pred_y) is np.ndarray:
            raise ValueError("No blackbox or predictions was set")

        self.X_ = X
        self.y_ = y
        if not type(pred_y) is np.ndarray:
            pred_y = self.black_box.predict(X)
        if X.getformat() == 'csr':
            X = X.toarray()
        if len(y.shape) == 1:
            y = to_categorical(y, len(self.classes_))
        if len(pred_y.shape) == 1:
            pred_y = to_categorical(pred_y, len(self.classes_))
        self.wrapper_ = DirichletUncertaintyWrapper(lambda_reg=self.lambda_reg,
                                                    num_samples=self.num_samples,
                                                    learning_rate=self.learning_rate,
                                                    num_hidden_units=self.num_hidden_units, verbose=self.verbose)
        self.wrapper_.train_model(X, y, pred_y, batch_size=self.batch_size, epochs=self.epochs)
        # Return the classifier
        return self

    def predict(self, X, pred_y = None):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        pred_y : array-like, shape (n_samples,)
            Array of predictions corresponding to calls to the blackbox for X.
            It can be used when no access to the representation of the input for the blackbox.
            An array of int.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The original prediction together with the corresponding uncertainty for each prediction.
        """
        # Input validation
        X = check_array(X, accept_sparse=True)

        if not type(pred_y) is np.ndarray:
            pred_y = self.black_box.predict(X)
        if X.getformat() == 'csr':
            X = X.toarray()
        if len(pred_y.shape) == 1:
            pred_y = to_categorical(pred_y, len(self.classes_))
        uncertainties = K.get_session().run(self.wrapper_.predict_entropy(X=X, pred_y=pred_y))
        result = np.array((np.argmax(pred_y, axis=1), uncertainties))
        return result.T
