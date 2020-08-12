from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.constraints import non_neg, Constraint
from tensorflow.keras import backend as K

import numpy as np

class SequenceBuilder(Sequence):
    """
    Generate Batches of data with appropriate fields (including numeric ones if any)
    Process time-to-event field if use_time is true
    Pad sequence of codes (up to maxlen in data) per visit
    Pad sequence of visits up to max_seq_len long per patient
    """

    def __init__(self, data, ARGS, target, model_parameters=None, target_out=False):
        # Extract codes
        self.codes = data[0]
        self.target = target
        self.batch_size = ARGS.batch_size
        self.target_out = target_out
        self.num_codes = ARGS.num_codes if 'num_codes' in vars(ARGS) else (model_parameters.num_codes if model_parameters.num_codes else -99) #used as pad value
        self.numeric_size = ARGS.numeric_size if 'numeric_size' in vars(ARGS) else (model_parameters.numeric_size if model_parameters.numeric_size else 0)
        self.use_time = ARGS.use_time if 'use_time' in vars(ARGS) else (model_parameters.use_time if model_parameters.use_time else False)
        self.max_seq_len = ARGS.max_seq_len if 'max_seq_len' in vars(ARGS) else 1000000
        # self.balance = (1-(float(sum(target))/len(target)))/(float(sum(target))/len(target))

        index = 1
        # Extract additional numeric fields if any
        if self.numeric_size > 0:
            self.numeric = data[index]
            index += 1
        # if time-to-event data
        if self.use_time:
            self.time = data[index]


    def __len__(self):
        """Compute number of batches.
        Add extra batch if the data doesn't exactly divide into batches
        """
        if len(self.codes) % self.batch_size == 0:
            return len(self.codes) // self.batch_size
        return len(self.codes) // self.batch_size + 1

    def __getitem__(self, idx):
        """Get batch of specific index"""
        def pad_data(data, length_visits, length_codes, pad_value=0):
            """Pad data to desired number of visits and codes inside each visit"""
            zeros = np.full((len(data), length_visits, length_codes), pad_value)
            for steps, mat in zip(data, zeros):
                if steps != [[-1]]:
                    for step, mhot in zip(steps, mat[-len(steps):]):
                        # Populate the data into the appropriate visit
                        mhot[:len(step)] = step
            return zeros

        # Compute reusable batch slice
        batch_slice = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        x_codes = self.codes[batch_slice]

        # Max number of visits and codes inside the visit for this batch
        pad_length_visits = min(max(map(len, x_codes)), self.max_seq_len)
        pad_length_codes = max(map(lambda x: max(map(len, x)), x_codes))
        # Number of elements in a batch (useful in case of partial batches)
        length_batch = len(x_codes)
        # Pad data
        x_codes = pad_data(x_codes, pad_length_visits, pad_length_codes, pad_value=self.num_codes)
        outputs = [x_codes]

        # Add numeric data if necessary
        if self.numeric_size > 0:
            x_numeric = self.numeric[batch_slice]
            x_numeric = pad_data(x_numeric, pad_length_visits, self.numeric_size, -99.0)
            outputs.append(x_numeric)
        # Add time data if necessary
        if self.use_time:
            x_time = (pad_sequences(self.time[batch_slice],
                                   dtype=np.float32, maxlen=pad_length_visits,value=+99)
                      .reshape(length_batch, pad_length_visits, 1)
                      )
            outputs.append(x_time)

        # Add target if necessary (training vs validation)
        if self.target_out:
            target = self.target[batch_slice].reshape(length_batch, 1, 1)
            # sample_weights = (target*(self.balance-1)+1).reshape(length_batch, 1)
            # In our experiments sample weights provided worse results
            return (outputs, target)

        return outputs


# BUG: embeddings_constraint not supported so no point freezing padding
# https://github.com/tensorflow/tensorflow/issues/33755
class FreezePadding_Non_Negative(Constraint):
    """Freezes the last weight to be near 0 and prevents non-negative embeddings"""
    def __call__(self, w):
        other_weights = K.cast(K.greater_equal(w, 0)[:-1], K.floatx())
        last_weight = K.cast(K.equal(K.reshape(w[-1, :], (1, K.shape(w)[1])), 0.), K.floatx())
        appended = K.concatenate([other_weights, last_weight], axis=0)
        w *= appended
        return w


class FreezePadding(Constraint):
    """Freezes the last weight to be near 0."""
    def __call__(self, w):
        other_weights = K.cast(K.ones(K.shape(w))[:-1], K.floatx())
        last_weight = K.cast(K.equal(K.reshape(w[-1, :], (1, K.shape(w)[1])), 0.), K.floatx())
        appended = K.concatenate([other_weights, last_weight], axis=0)
        w *= appended
        return w
