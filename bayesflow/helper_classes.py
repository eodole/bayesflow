# Copyright 2022 The BayesFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
import numpy as np
import pandas as pd
import tensorflow as tf

import logging
logging.basicConfig()

from sklearn.linear_model import HuberRegressor

from bayesflow.default_settings import DEFAULT_KEYS

class SimulationDataset:
    """ Helper class to create a tensorflow Dataset which returns
    dictionaries in BayesFlow format.
    """
    
    def __init__(self, forward_dict, batch_size):
        """
        Create a tensorfow Dataset from forward inference outputs and determines format. 
        """
        
        slices, keys_used, keys_none, n_sim = self._determine_slices(forward_dict)
        self.data = tf.data.Dataset\
                .from_tensor_slices(tuple(slices))\
                .shuffle(n_sim)\
                .batch(batch_size)
        self.keys_used = keys_used
        self.keys_none = keys_none
        self.n_sim = n_sim
        
    def _determine_slices(self, forward_dict):
        """ Determine slices for a tensorflow Dataset.
        """
        
        keys_used = []
        keys_none = []
        slices = []
        for k, v in forward_dict.items():
            if forward_dict[k] is not None:
                slices.append(v)
                keys_used.append(k)
            else:
                keys_none.append(k)
        n_sim = forward_dict[DEFAULT_KEYS['sim_data']].shape[0]
        return slices, keys_used, keys_none, n_sim
    
    def __call__(self, batch_in):
        """ Convert output of tensorflow Dataset to dict.
        """
        
        forward_dict = {}
        for key_used, batch_stuff in zip(self.keys_used, batch_in):
            forward_dict[key_used] = batch_stuff.numpy()
        for key_none in zip(self.keys_none):
            forward_dict[key_none] = None
        return forward_dict
    
    def __iter__(self):
        return map(self, self.data)


class RegressionLRAdjuster:
    """This class will compute the slope of the loss trajectory and inform learning rate decay."""
    
    def __init__(self, optimizer, period=100, wait_between_fits=10, patience=8, tolerance=-0.1, 
                 reduction_factor=0.25, num_resets=4, **kwargs):
        """ Creates an instance with given hyperparameters which will track the slope of the 
        loss trajectory according to specified hyperparameters and then issue an optional
        stopping suggestion.
        
        Parameters
        ----------

        optimizer         : tf.keras.optimizers.Optimizer instance
            An optimizer implementing a lr() method
        period            : int, optional, default: 100
            How much loss values to consider from the past
        wait_between_fits : int, optional, default: 10
            How many backpropagation updates to wait between two successive fits
        patience          : int, optional, default: 8
            How many successive times the tolerance value is reached before lr update.
        tolerance         : float, optional, default: -0.1
            The minimum slope to be considered substantial for training.
        reduction_factor  : float in [0, 1], optional, default: 0.25
            The factor by which the learning rate is reduced upon hitting the `tolerance`
            threshold for `patience` number of times
        num_resets        : int, optional, default: 4
            How many times to reduce the learning rate before issuing an optional stopping
        **kwargs          : dict, optional, default {}
            Additional keyword arguments passed to the `HuberRegression` class.
        """
        
        self.optimizer = optimizer
        self.period = period
        self.wait_between_periods = wait_between_fits
        self.regressor = HuberRegressor(**kwargs)
        self.t_vector = np.linspace(0, 1, self.period)[:, np.newaxis]
        self.patience = patience
        self.tolerance = tolerance
        self.num_resets = num_resets
        self.reduction_factor = reduction_factor
        self._reset_counter = 0
        self._patience_counter = 0
        self._cooldown_counter = 0
        self._wait_counter = 0
        self._slope = None
        self._is_waiting = False
        self._in_cooldown = False
        
    def get_slope(self, losses):
        """ Fits a Huber regression on the provided loss trajectory or returns None if
        not enough data points present.
        """
        
        # Return None if not enough loss values present
        if losses.shape[0] < self.period:
            return None

        if self._in_cooldown:
            self._cooldown_counter += 1
        
        # Check if still in a waiting phase and return old slope
        # if still waiting, otherwise refit Huber regression
        wait = self._check_waiting()
        if wait:
            return self._slope
        else:
            self.regressor.fit(self.t_vector, losses[-self.period:])
            self._slope = self.regressor.coef_[0]
            self._check_patience()
            return self._slope

    def _check_patience(self):
        """ Determines whether to reduce learning rate or be patient."""

        # Do nothing, if still in cooldown period
        if self._in_cooldown and self._cooldown_counter < self.period:
            return 
        # Otherwise set cooldown flag to False and reset counter
        else:
            self._in_cooldown = False
            self._cooldown_counter = 0

        # Check if negetaive slope too small
        if self._slope > self.tolerance:
            self._patience_counter += 1
        else:
            self._patience_counter = max(0, self._patience_counter - 1)

        # Check if patience surpassed and issue a reduction in learning rate
        if self._patience_counter >= self.patience:
            self._reduce_learning_rate()
            self._patience_counter = 0

    def _reduce_learning_rate(self):
        """ Reduces the learning rate by a given factor. """

        # Logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        if self._reset_counter >= self.num_resets:
            # TODO - add actual functionality
            logging.info('Triggered optional stopping!')
        else:
            # Take care of updating learning rate
            old_lr = self.optimizer.lr.numpy()
            new_lr = self.reduction_factor * old_lr
            self.optimizer.lr.assign(new_lr)
            self._reset_counter += 1

            # Verbose info to user
            logger.info(f'Reducing learning rate from {old_lr} to: {new_lr}...')

            # Set cooldown flag to avoid reset for some time.
            self._in_cooldown = True

    def _check_waiting(self):
        """ Determines whether to compute a new slope or wait."""

        # Case currently waiting
        if self._is_waiting:
            # Case currently waiting but period is over
            if self._wait_counter >= self.wait_between_periods - 1:
                self._wait_counter = 0
                self._is_waiting = False
            # Case currently waiting and period not over
            else:
                self._wait_counter += 1
            return True
        # Case not waiting
        else:
            self._is_waiting = True
            self._wait_counter += 1
            return False


class LossHistory:
    """ Helper class to keep track of losses during training.
    """
    def __init__(self):
        self.history = {}
        self.loss_names = []
        self._current_run = 0
        self._total_loss = []

    @property
    def total_loss(self):
        return np.array(self._total_loss)

    def start_new_run(self):
        self._current_run += 1
        self.history[f'Run {self._current_run}'] = {}

    def add_entry(self, epoch, current_loss):
        """ Adds loss entry for current epoch into internal memory data structure.
        """

        # Add epoch key, if specified
        if self.history[f'Run {self._current_run}'].get(f'Epoch {epoch}') is None:
            self.history[f'Run {self._current_run}'][f'Epoch {epoch}'] = []

        # Handle dict loss output
        if type(current_loss) is dict:
            # Store keys, if none existing
            if self.loss_names == []:
                self.loss_names = [k for k in current_loss.keys()]
            
            # Create and store entry
            entry = [v.numpy() if type(v) is not np.ndarray else v for v in current_loss.values()]
            self.history[f'Run {self._current_run}'][f'Epoch {epoch}'].append(entry)

            # Add entry to total loss
            self._total_loss.append(sum(entry))

        # Handle tuple or list loss output
        elif type(current_loss) is tuple or type(current_loss) is list:
            entry = [v.numpy() if type(v) is not np.ndarray else v for v in current_loss]
            self.history[f'Run {self._current_run}'][f'Epoch {epoch}'].append(entry)
            # Store keys, if none existing
            if self.loss_names == []:
                self.loss_names = [f'Loss.{l}' for l in range(1, len(entry)+1)]

            # Add entry to total loss
            self._total_loss.append(sum(entry))
        
        # Assume scalar loss output
        else:
            self.history[f'Run {self._current_run}'][f'Epoch {epoch}'].append(current_loss.numpy())
            # Store keys, if none existing
            if self.loss_names == []:
                self.loss_names.append('Default.Loss')
            
            # Add entry to total loss
            self._total_loss.append(current_loss.numpy())

    def get_running_losses(self, epoch):
        """ Compute and return running means of the losses for current epoch.
        """

        means = np.atleast_1d(np.mean(self.history[f'Run {self._current_run}'][f'Epoch {epoch}'], axis=0))
        if means.shape[0] == 1:    
            return {'Avg.Loss': means[0]}
        else:
            return {'Avg.' + k: v for k, v in zip(self.loss_names, means)}

    def get_plottable(self):
        """ Returns the losses as a nicely formatted pandas DataFrame.
        """

        # Get losses
        losses_df = pd.DataFrame(pd.concat([pd.melt(pd.DataFrame(self.history[r])) for r in self.history], axis=0).value.to_list())
        losses_df.columns = self.loss_names
        return losses_df.copy()

    def flush(self):
        """ Returns current history and removes all existing loss history.
        """

        h = self.history
        self.history = {}
        self._current_run = 0
        return h

    def get_copy(self):
        return deepcopy(self.history)


class SimulationMemory:
    
    def __init__(self, stores_raw=True, capacity_in_batches=50):
        self.stores_raw = stores_raw
        self._capacity = capacity_in_batches
        self._buffer = [None] * self._capacity
        self._idx = 0
        self.size_in_batches = 0

    def store(self, forward_dict):
        """ Stores simulation outputs, if internal buffer is not full.

        Parameters
        ----------
        forward_dict : dict
            The configured outputs of the forward model.
        """

        # If full, overwrite at index
        if not self.is_full():
            self._buffer[self._idx] = forward_dict
            self._idx += 1
            self.size_in_batches += 1
    
    def get_memory(self):
        return deepcopy(self._buffer)

    def is_full(self):
        """ Returns True if the buffer is full, otherwis False."""

        if self._idx >= self._capacity:
            return True
        return False