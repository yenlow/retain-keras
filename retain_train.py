"""Implementation of RETAIN Keras from Edward Choi"""
import os, argparse
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import non_neg
# from keras_exp.multigpu import get_available_gpus, make_parallel
from tensorflow.keras.metrics import AUC, Recall, Precision, SpecificityAtSensitivity, SensitivityAtSpecificity

from classes.io import read_data
from classes.sequences import SequenceBuilder, FreezePadding, FreezePadding_Non_Negative
#from classes.callbacks import LogEval
from classes.metrics import specificity

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', 500)

def parse_arguments(parser):
    """Read user arguments"""
    # 4895 is the voccabulary size in MIMIC-III
    parser.add_argument('--num_codes', type=int, default=4895,
                        help='Number of medical codes')
    parser.add_argument('--numeric_size', type=int, default=0,
                        help='Number of numeric input features, 0 if none')
    parser.add_argument('--use_time', action='store_true',
                        help='If argument is present the time input will be used')
    parser.add_argument('--emb_size', type=int, default=256,
                        help='Dimension of the visit embedding')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--max_seq_len', type=int, default=300,
                        help='Maximum number of visits after which the data is truncated\
                        (think max sequence length for padding)')
    parser.add_argument('--recurrent_size', type=int, default=256,
                        help='Dimension of hidden recurrent layers')
    parser.add_argument('--path_data_train', type=str, default='data/data_train.pkl',
                        help='Path to train data')
    parser.add_argument('--path_data_test', type=str, default='data/data_test.pkl',
                        help='Path to test data')
    parser.add_argument('--path_target_train', type=str, default='data/target_train.pkl',
                        help='Path to train target')
    parser.add_argument('--path_target_test', type=str, default='data/target_test.pkl',
                        help='Path to test target')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch Size')
    parser.add_argument('--dropout_input', type=float, default=0.4,
                        help='Dropout rate for embedding')
    parser.add_argument('--dropout_context', type=float, default=0.6,
                        help='Dropout rate for context vector')
    parser.add_argument('--l2', type=float, default=0.0001,
                        help='L2 regularization value')
    parser.add_argument('--out_directory', type=str, default='Model',
                        help='out_directory to save the model and the log file to')
    parser.add_argument('--allow_negative', action='store_true',
                        help='If argument is present the negative weights for embeddings/attentions\
                         will be allowed (original RETAIN implementation)')
    args = parser.parse_known_args()[0]

    return args


def model_create(ARGS):
    """Create and Compile model and assign it to provided devices"""

    def retain(ARGS):
        """Create the model"""

        # Define the constant for model saving
        reshape_size = ARGS.emb_size + ARGS.numeric_size
        if ARGS.allow_negative:
            embeddings_constraint = FreezePadding()
            beta_activation = 'tanh'
            output_constraint = None
        else:
            embeddings_constraint = FreezePadding_Non_Negative()
            beta_activation = 'sigmoid'
            output_constraint = non_neg()

        def reshape(data):
            """Reshape the context vectors to 3D vector"""
            return K.reshape(x=data, shape=(K.shape(data)[0], 1, reshape_size))

        # Code Input
        codes = L.Input((None, None), name='codes_input')
        inputs_list = [codes]
        # Calculate embedding for each code and sum them to a visit level
        codes_embs_total = L.Embedding(ARGS.num_codes + 1,
                                       ARGS.emb_size,
                                       name='embedding'
                                       # BUG: embeddings_constraint not supported
                                       # https://github.com/tensorflow/tensorflow/issues/33755
                                       #                                       ,embeddings_constraint=embeddings_constraint
                                       )(codes)
        codes_embs = L.Lambda(lambda x: K.sum(x, axis=2))(codes_embs_total)
        # Numeric input if needed
        if ARGS.numeric_size > 0:
            numerics = L.Input((None, ARGS.numeric_size), name='numeric_input')
            inputs_list.append(numerics)
            full_embs = L.concatenate([codes_embs, numerics], name='catInp')
        else:
            full_embs = codes_embs

        # Apply dropout on inputs
        full_embs = L.Dropout(ARGS.dropout_input)(full_embs)

        # Time input if needed
        if ARGS.use_time:
            time = L.Input((None, 1), name='time_input')
            inputs_list.append(time)
            time_embs = L.concatenate([full_embs, time], name='catInp2')
        else:
            time_embs = full_embs

        # Setup Layers
        # This implementation uses Bidirectional LSTM instead of reverse order
        #    (see https://github.com/mp2893/retain/issues/3 for more details)

        # If training on GPU and Tensorflow use CuDNNLSTM for much faster training
        if glist:
            alpha = L.Bidirectional(L.CuDNNLSTM(ARGS.recurrent_size, return_sequences=True),
                                    name='alpha')
            beta = L.Bidirectional(L.CuDNNLSTM(ARGS.recurrent_size, return_sequences=True),
                                   name='beta')
        else:
            alpha = L.Bidirectional(L.LSTM(ARGS.recurrent_size,
                                           return_sequences=True, implementation=2),
                                    name='alpha')
            beta = L.Bidirectional(L.LSTM(ARGS.recurrent_size,
                                          return_sequences=True, implementation=2),
                                   name='beta')

        alpha_dense = L.Dense(1, kernel_regularizer=l2(ARGS.l2))
        beta_dense = L.Dense(ARGS.emb_size + ARGS.numeric_size,
                             activation=beta_activation, kernel_regularizer=l2(ARGS.l2))

        # Compute alpha, visit attention
        alpha_out = alpha(time_embs)
        alpha_out = L.TimeDistributed(alpha_dense, name='alpha_dense_0')(alpha_out)
        alpha_out = L.Softmax(axis=1)(alpha_out)
        # Compute beta, codes attention
        beta_out = beta(time_embs)
        beta_out = L.TimeDistributed(beta_dense, name='beta_dense_0')(beta_out)
        # Compute context vector based on attentions and embeddings
        c_t = L.Multiply()([alpha_out, beta_out, full_embs])
        c_t = L.Lambda(lambda x: K.sum(x, axis=1))(c_t)
        # Reshape to 3d vector for consistency between Many to Many and Many to One implementations
        contexts = L.Lambda(reshape)(c_t)

        # Make a prediction
        contexts = L.Dropout(ARGS.dropout_context)(contexts)
        output_layer = L.Dense(1, activation='sigmoid', name='dOut',
                               kernel_regularizer=l2(ARGS.l2), kernel_constraint=output_constraint)

        # TimeDistributed is used for consistency
        # between Many to Many and Many to One implementations
        output = L.TimeDistributed(output_layer, name='time_distributed_out')(contexts)
        # Define the model with appropriate inputs
        model = Model(inputs=inputs_list, outputs=[output])

        return model

    # Set Tensorflow to grow GPU memory consumption instead of grabbing all of it at once
    # TODO: create a tf session block
    K.clear_session()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    #    config.gpu_options.allow_growth = True
    tfsess = tf.Session(config=config)
    K.set_session(tfsess)
    # If there are multiple GPUs set up a multi-gpu model
    # Get available gpus , returns empty list if none
    # glist = get_available_gpus()
    glist = []
    if len(glist) > 1:
        with tf.device('/cpu:0'):
            model = retain(ARGS)
        model_final = make_parallel(model, glist)
    else:
        model_final = retain(ARGS)

    # Compile the model - adamax has produced best results in our experiments
    model_final.compile(optimizer='adamax',
                        loss='binary_crossentropy',
                        #TODO: add AUPRC?
                        metrics=[Recall(), specificity,
                                 SpecificityAtSensitivity(0.5,3),
                                 SensitivityAtSpecificity(0.5, 3),
                                 'accuracy', AUC(), Precision()],
                        sample_weight_mode="temporal")
    return model_final


def create_callbacks(model, data, ARGS):
    """Create keras custom callback with checkpoint and logging"""

    # Create callbacks
    if not os.path.exists(ARGS.out_directory):
        os.makedirs(ARGS.out_directory)
    # log to Model/log.txt as specified by ARGS.out_directory
    checkpoint_cb = ModelCheckpoint(filepath=ARGS.out_directory + '/weights.{epoch:03d}.hdf5',
                                    verbose=2, save_best_only=True)
    earlystopping_cb = EarlyStopping(monitor='val_loss', patience=3)
    # Use builtin logger instead of LogEval
    csv_cb = CSVLogger(f'{ARGS.out_directory}/log.txt', append=False, separator='\t')
#    custom_callback = LogEval(f'{ARGS.out_directory}/log.txt', model, data, ARGS, interval=1, extlog=True)
    callback_list = [checkpoint_cb, earlystopping_cb, csv_cb]
    return callback_list


def train_model(model, data_train, y_train, data_test, y_test, ARGS):
    """Train the Model with appropriate callbacks and generator"""
    callback_list = create_callbacks(model, (data_test, y_test), ARGS)
    train_generator = SequenceBuilder(data_train, ARGS, target=y_train)
    test_generator = SequenceBuilder(data_test, ARGS, target=y_test)
    history = model.fit_generator(  generator=train_generator,
                                    epochs=ARGS.epochs, verbose=2,
                                    validation_data=test_generator,
                                    # validation_freq=[1, 5, 10],
                                    callbacks=callback_list
                                    # ,max_queue_size=15, use_multiprocessing=False,
                                    # workers=3, initial_epoch=0
                                    )
    return history


def main(ARGS):
    """Main function"""
    print('Reading Data')
    data_train, y_train, data_test, y_test = read_data(ARGS)

    print('Creating Model')
    model = model_create(ARGS)
    print(model.summary())

    print('Training Model')
    history = train_model(model=model, data_train=data_train, y_train=y_train,
                          data_test=data_test, y_test=y_test, ARGS=ARGS)
    # callback history metrics is streamed to Models/log.txt as specified in ARGS.out_directory/log.txt
    # no need to save the pd df
    print(pd.DataFrame(history.history))


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ARGS = parse_arguments(PARSER)
    main(ARGS)
