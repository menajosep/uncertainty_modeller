from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_accuracy
import tensorflow as tf
import tensorflow_probability as tfp


class DirichletUncertaintyWrapper(object):
    """ A wrapper that estimates the uncertainty of the predictions of a classifier by modeling each point
        as a Dirichlet distribution.

        For more information look at the paper:
        Mena, J., Pujol, O., & Vitrià, J. (2020).
        Uncertainty-based Rejection Wrappers for Black-box Classifiers. IEEE Access.

        Parameters
        ----------
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
        num_classes_ : int
            The number of classes to predict.
        model_ : object
            The model used for learning and estimating the uncertainties
        training_history_: object
            Training history
        """
    def __init__(self, lambda_reg=1e-2, num_samples=1000,
                 learning_rate=1e-3, num_hidden_units=20,
                 verbose=0):
        self.lambda_reg = lambda_reg
        self.epsilon = 1e-10
        self.num_samples = num_samples
        self.learning_rate = learning_rate
        self.num_hidden_units = num_hidden_units
        self.verbose = verbose

    def dirichlet_aleatoric_cross_entropy(self, y_true, y_pred):
        """
            Loss function that applies a categorical cross entropy to the predictions
            obtained from the original model combined with a beta parameter that models
            the aleatoric noise associated with each data point. We model a Dirichlet
            pdf based on the combination of both the prediction and the beta parameter
            to sample from it and obtain the resulting output probabilities

            Parameters
            ----------
            y_true:  `np.array`
                the labels in one hot encoding format

            y_pred: `np.array`
                output of the model formed by the concatenation of the original prediction
                in the first num_classes positions, and a beta scalar in the last one


            Returns
            -------
            an array with the associated cross-entropy for each element of the batch

        """
        # original predictions obtained with the blackbox
        mu_probs = y_pred[:, :self.num_classes_]
        # scale param estimated by the wrapper
        beta = y_pred[:, self.num_classes_:]
        # parameter of the Dirichlet distribution
        alpha = mu_probs * beta
        dirichlet = tfp.distributions.Dirichlet(alpha)
        # sample the Dir distribution for each point
        z = dirichlet.sample(sample_shape=self.num_samples)
        # compute the crossentropy for the mean of the samples
        e_probs = tf.reduce_mean(z, axis=0)
        log_probs = tf.log(e_probs + self.epsilon)
        cross_entropy = -(tf.reduce_sum(y_true * log_probs, axis=-1))
        # the loss includes a regularization term for constraining the beta to grow unbounded
        return cross_entropy + self.lambda_reg * tf.reduce_sum(beta, axis=-1)

    def max_beta(self, y_true, y_pred, **args):
        """
            metric that outputs the max value for the beta

            Parameters
            ----------
            y_true:  `np.array`
                the labels in one hot encoding format

            y_pred: `np.array`
                output of the model formed by the concatenation of the original prediction
                in the first num_classes positions, and a beta scalar in the last one


            Returns
            -------
            the max value for the beta

        """
        beta = y_pred[:, self.num_classes_:]
        return tf.reduce_max(beta)

    def min_beta(self, y_true, y_pred, **args):
        """
            metric that outputs the min value for the beta

            Parameters
            ----------
            y_true:  `np.array`
                the labels in one hot encoding format

            y_pred: `np.array`
                output of the model formed by the concatenation of the original prediction
                in the first num_classes positions, and a beta scalar in the last one


            Returns
            -------
            the min value for the beta

        """
        beta = y_pred[:, self.num_classes_:]
        return tf.reduce_min(beta)

    # metric that outputs the accuracy when only considering the logits_mu.
    # this accuracy should be the same that was obtained with the fake classifier
    # in its best epoch.
    # def mu_accuracy(self):
    #  num_classes = self.num_classes
    def mu_accuracy(self, y_true, y_pred, **args):
        """
            metric that outputs the accuracy when only considering the logits_mu.
            This accuracy should be the same that was obtained with the fake classifier in its best epoch.

            Parameters
            ----------
            y_true:  `np.array`
                the labels in one hot encoding format

            y_pred: `np.array`
                output of the model formed by the concatenation of the original prediction
                in the first num_classes positions, and a beta scalar in the last one


            Returns
            -------
            the accuracy of the original predictions

        """
        mu_probs = y_pred[:, :self.num_classes_]
        y_true_probs = y_true[:, :self.num_classes_]
        return categorical_accuracy(y_true_probs, mu_probs)

    def create_model(self, input_shape):
        """
            Method that creates the Keras Neural Network model for the DL  wrapper that will estimate the betas.

            Parameters
            ----------
            input_shape:  `int`
                shape of the input


            Returns
            -------
            a model to be trained

        """
        model_input = Input(shape=(input_shape,))
        logits_sigma = Dense(self.num_hidden_units, activation='relu')(model_input)
        logits_sigma = Dense(self.num_hidden_units, activation='relu')(logits_sigma)
        logits_sigma = Dense(self.num_hidden_units, activation='relu')(logits_sigma)
        logits_sigma = Dense(self.num_hidden_units, activation='relu')(logits_sigma)
        logits_sigma = Dense(1, activation='sigmoid')(logits_sigma)
        probs_mu = Input(shape=(self.num_classes_,))
        output = concatenate([probs_mu, logits_sigma])

        model = Model(inputs=[model_input, probs_mu], outputs=output)
        model.compile(loss=self.dirichlet_aleatoric_cross_entropy,
                      optimizer=Adam(lr=self.learning_rate),
                      metrics=[self.mu_accuracy, self.min_beta, self.max_beta])
        return model

    def train_model(self, X, y, pred_y, batch_size=256, epochs=50):
        """
            Method that trains the wrapper for the inputs, and the corresponding labels and predictions of the blackbox.

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
            batch_size : int, default='demo'
                Batch size used when training uncertainties.
            epochs : int, default='demo'
                Number of epochs when training the uncertainties.


            Returns
            -------
            None

        """
        self.num_classes_ = pred_y.shape[1]
        input_shape = X.shape[1]
        self.model_ = self.create_model(input_shape)
        self.training_history_ = self.model_.fit([X, pred_y],
                                                 y,
                                                 batch_size=batch_size,
                                                 epochs=epochs,
                                                 shuffle=True,
                                                 verbose=self.verbose,
                                                 validation_split=0.2)

    def predict_entropy(self, X, pred_y):
        """
            Method that trains the wrapper for the inputs, and the corresponding labels and predictions of the blackbox.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The training input samples.
            pred_y : array-like, shape (n_samples,)
                Array of predictions corresponding to calls to the blackbox for X.
                It can be used when no access to the representation of the input for the blackbox.
                An array of int.


            Returns
            -------
            An array with the uncertainty for each data input and prediction

        """
        y_pred = self.model_.predict([X, pred_y])
        # original predictions obtained with the blackbox
        mu_probs = y_pred[:, :self.num_classes_]
        # scale param estimated by the wrapper
        beta = y_pred[:, self.num_classes_:]
        # parameter of the Dirichlet distribution
        alpha = mu_probs * beta
        dirichlet = tfp.distributions.Dirichlet(alpha)
        # sample the Dir distribution for each point
        z = dirichlet.sample(sample_shape=self.num_samples)
        # compute the entropy between the original pred and the mean of the samples
        e_probs = tf.reduce_mean(z, axis=0)
        log_probs = tf.log(e_probs + self.epsilon)
        entropy = tf.reduce_sum(-e_probs * log_probs, axis=-1)
        return entropy
