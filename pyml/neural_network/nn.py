"""Neural network

This module provides a simple implementation of a neural network."""

from __future__ import annotations

# Import of parent classes
from pyml.neural_network.layer import _Layer
from pyml.neural_network.loss import _Loss
from pyml.neural_network.optimizer import _Optimizer
from pyml.neural_network.layer.activation import _Activation
from pyml.neural_network.layer.transformation import _Transformation, _TrainableTransformation, Input, Dropout
from pyml.utils.accuracy import _Accuracy

# Import layers, losses, optimizers
from pyml.neural_network.layer.activation import Softmax
from pyml.neural_network.loss import CategoricalCrossentropy, Softmax_CategoricalCrossentropy

# Exceptions
from pyml.exceptions import HyperparametersNotSpecified

import pickle
import copy
import os
import numpy as np

class InconsistentLayerSizes(Exception):
    """Exception raised when neurons of two consecutive layers do not match.

    Parameters
    ----------
    output_size_previous_layer : int
        Ouput neurons from previous layers
    input_size_current_layer : int
        Input size of current layer

    Examples
    --------
    >>> from pyml.nn import NN
    >>> model = NN()
    >>> model.add(Dense(4, 16))
    >>> model.add(Activation_ReLU())
    >>> model.add(Layer_Dense(20, 10))
    InconsistentLayerSizes: The output size of the previous layer: 16 and the input size of the current layer: 17 do not match.
    """

    def __init__(self, output_size_previous_layer:int, input_size_current_layer:int) -> None:
        self.message = f'The output size of the previous layer: {output_size_previous_layer} ' + \
            f'and the input size of the current layer: {input_size_current_layer} do not match.'
        super().__init__(self.message)

class NN():
    """Neural network class for regression & classification

    This class provides functionality for building, training, and evaluating neural networks.

    Attributes
    ----------
    layers : list[_Layer]
        A list of layers in the neural network.
    trainable_layers : list[_TrainableTransformation]
        A list of trainable layers in the neural network.
    loss : _Loss
        The loss function used for training the network.
    optimizer : _Optimizer
        The optimizer used for updating network parameters during training.
    accuracy : _Accuracy
        The accuracy metric used for evaluating the network's performance.
    finished_build : bool
        Indicates whether the neural network has been built.

    
    Examples
    --------
    >>> from pyml.nn import NN
    >>> network = NN()
    >>> network.add_layer(Dense(4, 512))
    >>> network.add_layer(ReLU())
    >>> network.add_layer(Dropout(0.2))
    >>> network.add_layer(Dense(512, NUM_CLASSES))
    >>> network.add_layer(Softmax())

    >>> model.set_loss(CategoricalCrossentropy())
    >>> model.set_optimizer(Adam(learning_rate=0.005, decay=5e-5))
    >>> model.set_accuracy(MultiClassAccuracy())

    >>> model.build()
    """

    def __init__(self):

        self.layers = []
        self.trainable_layers = []

        # Hyperparameters
        self.loss = None
        self.optimizer = None
        self.accuracy = None

        self.finished_build = False

        self.softmax_classifier_output = None


    def _get_last_connected_layer(self) -> None:
        
        for layer in reversed(self.layers):
            if isinstance(layer, _Transformation) and not isinstance(layer, Dropout):
                return layer
        
    def add_layer(self, layer: _Layer) -> None:
        """Adds layer to the model

        Iteratively adds layers to the model.
        These layers can be e.g. conventional dense layers or activation functions.
        Be aware: the order of appending the layers is crucial and matters.

        Parameters
        ----------
        layer : _Layer
            The layer to be added to the network.

        Raises
        ------
        InconsistentLayerSizes
            Raised when adding a layer where input size does not match the output size of the previous layer

        Examples
        --------
        >>> from pyml.nn import NN
        >>> model = NN()
        >>> model.add(Dense(4, 16))
        >>> model.add(ReLU())
        >>> model.add(Dense(16, 32))
        >>> model.add(ReLU())
        >>> ...
        """

        self.layers.append(layer)
        return

        # TODO: Update to handle convolutions and reshape layer

        if not self.layers or isinstance(layer, _Activation) or isinstance(layer, Dropout):
            self.layers.append(layer)
            return
        
        last_layer = self._get_last_connected_layer()
        if last_layer.output_size != layer.input_size:
            raise InconsistentLayerSizes(last_layer.output_size, layer.input_size)
        
        self.layers.append(layer)
    
    def set_loss(self, loss:_Loss) -> None:
        """Set the loss function for the neural network.

        Parameters
        ----------
        loss : _Loss
            The loss function to be used for training.
        """
        self.loss = loss
    
    def set_optimizer(self, optimizer:_Optimizer) -> None:
        """Set the optimizer for the neural network.

        Parameters
        ----------
        optimizer : _Optimizer
            The optimizer to be used for updating parameters.
        """
        self.optimizer = optimizer
    
    def set_accuracy(self, accuracy:_Accuracy) -> None:
        """Set the accuracy metric for the neural network.

        Parameters
        ----------
        accuracy : _Accuracy
            The accuracy metric to be used for evaluation.
        """
        self.accuracy = accuracy

    def check_hyperparameters(self) -> None:
        """Check if essential hyperparameters are specified.

        Raises
        ------
        HyperparametersNotSpecified
            If any of the essential hyperparameters is not specified.
        """
        if self.loss is None:
            raise HyperparametersNotSpecified('loss')
        elif self.optimizer is None:
            raise HyperparametersNotSpecified('optimizer')
        elif self.accuracy is None:
            raise HyperparametersNotSpecified('accuracy')

    def build(self) -> None:
        """Build the neural network architecture.

        This method sets up the connections between layers and prepares the network for training.
        """

        # Check that all hyperparameters are initialized
        self.check_hyperparameters()
        
        # Set input layer
        self.input_layer = Input()

        # Count the number of layers
        layer_count = len(self.layers)

        # Set order of the layers and specifiy their direct neighbors
        # and retrieve all trainable layers
        for i in range(layer_count):

            if i == 0:
                self.layers[i].set_adjacent_layers(
                    previous_layer = self.input_layer,
                    next_layer = self.layers[i+1]
                )
            elif i < layer_count - 1:
                self.layers[i].set_adjacent_layers(
                    previous_layer = self.layers[i-1],
                    next_layer = self.layers[i+1]
                )
            else:
                self.layers[i].set_adjacent_layers(
                    previous_layer = self.layers[i-1],
                    next_layer = self.loss
                )
                # Final layer is not the loss
                self.output_layer = self.layers[i]


            # Retrieve the trainable layers
            if isinstance(self.layers[i], _TrainableTransformation):
                self.trainable_layers.append(self.layers[i])

            
        # Pass the trainable layers to the loss instance
        self.loss.set_trainable_layers(self.trainable_layers)

        # Create a combined activation and loss function object with faster gradient calculation
        # if using Softmax output activation and Categorical Cross-Entropy loss function

        if isinstance(self.layers[-1], Softmax) and \
           isinstance(self.loss, CategoricalCrossentropy):
            self.softmax_classifier_output = \
                Softmax_CategoricalCrossentropy()


        self.finished_build = True

    @staticmethod
    def print_summary(
        context:str,
        accuracy:float,
        loss:float,
        data_loss:float=None,
        regularization_loss:float=None,
        learning_rate:float=None,
    ) -> None:
        """Print a summary of training or evaluation results.

        Parameters
        ----------
        context : str
            Context for the summary (e.g., 'training', 'validation').
        accuracy : float
            The accuracy achieved during training or evaluation.
        loss : float
            The total loss achieved during training or evaluation.
        data_loss : float, optional
            The data loss component of the total loss, by default None.
        regularization_loss : float, optional
            The regularization loss component of the total loss, by default None.
        learning_rate : float, optional
            The learning rate used during training, by default None.
        """
        
        # Set values to --- if not specified
        data_loss = '- - -' if data_loss is None else f'{data_loss:.3f}'
        regularization_loss = '- - -' if regularization_loss is None else f'{regularization_loss:.3f}'
        learning_rate = '- - -' if learning_rate is None else f'{learning_rate:.3f}'

        print(
            f'{context}, ' +
            f'acc: {accuracy:.3f}, ' +
            f'loss: {loss:.3f} (' +
            f'data_loss: {data_loss}, ' +
            f'reg_loss: {regularization_loss}), ' +
            f'lr: {learning_rate}'
        )


    def train(
        self,
        X:np.ndarray,
        y:np.ndarray,
        *,
        epochs:int=1,
        batch_size:int=None,
        validation_data:np.ndarray=None,
        verbose:int=0,
        print_summary_every:int=1,
        save_file_path:str=None
    ) -> None:
        """Train the neural network.

        Examples
        --------        
        >>> X_train, y_train, X_test, y_test = ...  # Load training data
        >>> network.train(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

        Parameters
        ----------
        X : numpy.ndarray
            The input training data.
        y : numpy.ndarray
            The target training data.
        epochs : int, optional
            The number of training epochs, by default 1.
        batch_size : int, optional
            The batch size for training, by default None.
        validation_data : numpy.ndarray, optional
            Validation data for evaluation during training.
            Should include data (X) and their labels (y), by default None.
        verbose : int, optional
            Verbosity level (0: no prints, 1: epoch summary, 2: detailed prints), by default 0.
        print_summary_every : int, optional
            Print summary every 'print_summary_every' steps, by default 1.
        save_file_path : str, optional
            Path to save the model parameters after training, by default None.
            TODO: NOT IMPLEMENTED YET
        """

        # VERBOSE DESCRIPTION------------------------|
        # VERBOSE == 0: no prints at all             |
        # VERBOSE == 1: print only epoch summary     |
        # VERBOSE == 2: print also after major steps |
        # -------------------------------------------|
    

        # Check if model has already been build, if not build now
        if not self.finished_build:
            self.build()
            if verbose>1: print('training: model had to been build')

        # Check that all hyperparameters are initialized
        self.check_hyperparameters()
        if verbose>1: print(f'training: all hyperparameters have been initialized')

        # Initialize accuracy instance
        self.accuracy.init(y)
        if verbose>1: print(f'training: accuracy instance has been initialized')

        # Compute step size
        # Training step size describes how many many batches will be forwarded during each epoch
        training_step_size = 1
        if batch_size is not None:
            # training step size = ⌊ count of tranings data / batch size ⌋
            training_step_size = len(X) // batch_size
            # Add single training step if some trainings data will remain left over
            if training_step_size * batch_size < len(X):
                training_step_size += 1
        if verbose>1: print(f'training: training steps size has been computed -- training step size: {training_step_size}, batch size: {batch_size}')

        # Start training
        if verbose>1: print(f'training: start training')

        for epoch in range(epochs):
            
            # Print current epoch
            if verbose>0: print(f'training: Epoch: {epoch}/{epochs}')

            # Reset loss and accuracy
            self.loss.reset()
            self.accuracy.reset()

            for training_step in range(training_step_size):

                # Print current training step
                if verbose>1: print(f'training: Training step: {training_step}/{training_step_size}')

                # Compute the batch
                if batch_size is None:
                    X_batch = X
                    y_batch = y
                else:
                    lower_limit = training_step*batch_size
                    upper_limit = (training_step+1)*batch_size
                    X_batch = X[lower_limit:upper_limit]
                    y_batch = y[lower_limit:upper_limit]
                
                # Compute forward pass
                if verbose>1: print(f'training: Compute forward pass')
                output = self.forward(X_batch, training=True)

                # Compute loss
                if verbose>1: print(f'training: Compute loss')
                data_loss, regularization_loss = self.loss.calculate(
                    output, 
                    y_batch, 
                    include_regularization=True
                )
                # Combine loss of data loss and regularization loss
                loss = data_loss + regularization_loss

                # Compute backward
                if verbose>1: print(f'training: Compute backward')
                self.backward(output, y_batch)

                # Update parameter weights
                if verbose>1: print(f'training: Update parameters')
                self.optimizer.pre_update_parameters()
                for layer in self.trainable_layers:
                    self.optimizer.update_parameters(layer)
                self.optimizer.post_update_parameters()

                # Calculate accuracy
                predictions = self.output_layer.predictions(output)
                accuracy = self.accuracy.calculate(predictions, y_batch)

                # Print step size summary
                if verbose>1 and (not training_step % print_summary_every or training_step == training_step_size - 1):
                    NN.print_summary(
                        'training',
                        accuracy,
                        loss,
                        data_loss,
                        regularization_loss,
                        self.optimizer.current_learning_rate
                    ) 

            # Print epoch summary
            epoch_data_loss, epoch_regularization_loss = \
                self.loss.calculate_accumulated(
                    include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            if verbose > 0:
                NN.print_summary(
                    'epoch',
                    epoch_accuracy,
                    epoch_loss,
                    epoch_data_loss,
                    epoch_regularization_loss,
                    self.optimizer.current_learning_rate
                )

            # Evaluate model on validaten data
            if validation_data is not None:
                self.evaluate(*validation_data, batch_size=batch_size, verbose=verbose)

        if verbose>1: print(f'training: Training finished')

    def _forward(self, layer:_Layer, X:np.ndarray, training:bool=False) -> None:
        """Perform a forward pass through a layer.

        If the layer is a dropout layer, the training context (true or false is also passed).

        Parameters
        ----------
        layer : _Layer
            The layer to perform the forward pass on.
        X : numpy.ndarray
            The input data.
        training : bool, optional
            Indicates if the network is in training mode, by default False.
        """
        if isinstance(layer, Dropout):
            layer.forward(X, training)
        else:
            layer.forward(X)
                
    def forward(self, X:np.ndarray, training:bool=False) -> np.ndarray:
        """Perform a forward pass through the entire neural network.

        Parameters
        ----------
        X : numpy.ndarray
            The input data.
        training : bool, optional
            Indicates if the network is in training mode, by default False.

        Returns
        -------
        numpy.ndarray
            The output of the network.
        """

        # Forward data trough input layer
        self._forward(self.input_layer, X, training)

        # Iterate through remaining layers
        for layer in self.layers:
            self._forward(layer, layer.previous_layer.output, training)

        # Return output of last layer
        return layer.output

    def backward(self, output:np.ndarray, y:np.ndarray) -> None:
        """Perform a backward pass through the neural network.

        Parameters
        ----------
        output : numpy.ndarray
            The output of the network.
        y : numpy.ndarray
            The target data.
        """
        # Conduct backward pass if softmax classifier
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
        
            # Skip backward step for last layer since we combined activation and loss
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next_layer.dinputs)
            
            # Finish
            return
        
        # Compute backward as usual
        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next_layer.dinputs)

    def evaluate(self, X_val:np.ndarray, y_val:np.ndarray, *, batch_size:int=None, verbose:int=0) -> None:
        """Evaluate the neural network on validation data.

        Parameters
        ----------
        X_val : numpy.ndarray
            The input validation data.
        y_val : numpy.ndarray
            The target validation data.
        batch_size : int, optional
            The batch size for evaluation, by default None.
        verbose : int, optional
            Verbosity level (0: no prints, 1: summary prints), by default 0.
        """

        # Default step size
        validation_step_size = 1
        # Calculate number of batches
        if batch_size is not None:
            # training step size = ⌊ count of tranings data / batch size ⌋
            training_step_size = len(X_val) // batch_size
            # Add single training step if some trainings data will remain left over
            if training_step_size * batch_size < len(X_val):
                training_step_size += 1

        # Reset loss and accuracy
        self.loss.reset()
        self.accuracy.reset()


        # Iterate over steps
        for validation_step in range(validation_step_size):

            # Compute the batch
            if batch_size is None:
                X_batch = X_val
                y_batch = y_val
            else:
                lower_limit = validation_step*batch_size
                upper_limit = (validation_step+1)*batch_size
                X_batch = X_val[lower_limit:upper_limit]
                y_batch = y_val[lower_limit:upper_limit]

            # Perform the forward pass
            output = self.forward(X_batch, training=False)

            # Calculate the loss
            self.loss.calculate(output, y_batch)

            # Get predictions and calculate an accuracy
            predictions = self.output_layer.predictions(output)
            self.accuracy.calculate(predictions, y_batch)

        # Retrieve validation loss and accuracy
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        # Print a summary
        if verbose > 0:
            NN.print_summary(
            'validation',
            validation_accuracy,
            validation_loss
            )


    def predict(self, X:np.ndarray, *, batch_size:int=None) -> np.ndarray:
        """Generate predictions using the trained neural network.

        Parameters
        ----------
        X : np.ndarray
            The input data for which predictions are to be generated.
        batch_size : int, optional
            The batch size for prediction, by default None.

        Returns
        -------
        numpy.ndarray
            The predictions generated by the network.
        """
        
        # Default value if batch size is not being set
        prediction_step_size = 1
        # Calculate number of batches
        if batch_size is not None:
            # training step size = ⌊ count of tranings data / batch size ⌋
            prediction_step_size = len(X) // batch_size
            # Add single training step if some trainings data will remain left over
            if prediction_step_size * batch_size < len(X):
                prediction_step_size += 1

        # Model output
        output = []

        for prediction_step in range(prediction_step_size):

            # Get batch
            if batch_size is None:
                batch_X = X
            else:
                lower_limit = prediction_step*batch_size
                upper_limit = (prediction_step+1)*batch_size
                batch_X = X[lower_limit:upper_limit]

        # Perform the forward pass
        batch_output = self.forward(batch_X, training=False)

        # Append batch prediction to the list of predictions
        output.append(batch_output)

        # Stack output
        output = np.vstack(output)
        return output

    def get_model_parameters(self) -> list[np.ndarray]:
        """_summary_

        Returns
        -------
        list[numpy.ndarray]
            A list of numpy arrays containing the parameters of each trainable layer.
        """
        parameters = []
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())

        return parameters
    
    def set_model_parameters(self, parameters:list[tuple[np.ndarray]]) -> None:
        """Set the parameters of the trainable layers in the model.

        Parameters
        ----------
        parameters : list[tuple[numpy.ndarray]]
            A list storing parameters for each trainable layer within a tuple.
        """
        for parameter, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter)

    def save_model_parameters(self, path:str) -> None:
        """Save the model parameters to a file.

        Parameters
        ----------
        path : str
            The path to the file where the parameters will be saved.
        """
        with open(path, 'wb') as f:
            pickle.dumb(self.get_model_parameters())

    def load_model_parameters(self, path:str) -> None:
        """Load and set the model parameters from a file.

        Parameters
        ----------
        path : str
            The path to the file from which the parameters will be loaded.
        """
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    def save_model(self, path:str) -> None:
        """Save the entire trained model to a file.

        Parameters
        ----------
        path : str
            The path to the file where the model will be saved.

        Examples
        --------
        >>> network.save_model('trained_model.model')
        """

        # make copy of current model
        model_copy = copy.deepcopy(self)

        # Reset loss and accuracy
        model_copy.loss.reset()
        model_copy.accuracy.reset()

        # Remove data related attributes in model
        model_copy.input_layer.__dict__.pop('output', None)
        model_copy.loss.__dict__.pop('dinputs', None)

        for layer in model_copy.layers:
            for property in ['inputs', 'output', 'dinputs',
                             'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)

        # Save model
        with open(path, 'wb') as f:
            pickle.dumb(model_copy, f)

    @staticmethod
    def load(path:str) -> NN:
        """Load a trained model from a file.

        Parameters
        ----------
        path : str
            The path to the file from which the model will be loaded.

        Returns
        -------
        NN
            The loaded trained model.

        Examples
        --------
        >>> loaded_model = NN.load('trained_model.model')
        """

        with open(path, 'rb') as f:
            model = pickle.load(f)

        return model
