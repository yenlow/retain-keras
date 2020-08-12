import pandas as pd
import pickle


def read_data(ARGS, model_parameters=None, mode='train'):
    # TODO: combine read_data in train and evaluate
    """Read the data from provided paths and assign it into lists"""
    if mode == 'train':
        data_train_df = pd.read_pickle(ARGS.path_data_train)
        data_test_df = pd.read_pickle(ARGS.path_data_test)
        y_train = pd.read_pickle(ARGS.path_target_train)['target'].values
        y_test = pd.read_pickle(ARGS.path_target_test)['target'].values
        data_output_train = [data_train_df['codes'].values]
        data_output_test = [data_test_df['codes'].values]

        if ARGS.numeric_size > 0:
            data_output_train.append(data_train_df['numerics'].values)
            data_output_test.append(data_test_df['numerics'].values)
        if ARGS.use_time:
            data_output_train.append(data_train_df['to_event'].values)
            data_output_test.append(data_test_df['to_event'].values)

        output = (data_output_train, y_train, data_output_test, y_test)

    else:   #not training
        data = pd.read_pickle(ARGS.path_data)
        data_output = [data['codes'].values]

        if not model_parameters:
            raise ValueError("model_parameters must not be None when mode = 'eval'")
        else:
            if model_parameters.numeric_size:
                    data_output.append(data['numerics'].values)
            if model_parameters.use_time:
                data_output.append(data['to_event'].values)

        # only when evaluating with ground truth
        if 'path_target' in vars(ARGS):
            y = pd.read_pickle(ARGS.path_target)['target'].values
            output = (data_output, y)
        else:
            output = data_output

    return output


def read_dictionary(ARGS, padding=''):
    with open(ARGS.path_dictionary, 'rb') as f:
        dictionary = pickle.load(f)
    dictionary[len(dictionary)] = padding  #2nd last value
    return dictionary

