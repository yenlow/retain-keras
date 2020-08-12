"""This function will load the given data and continuosly interpet selected patients"""
import argparse
import pandas as pd
from tensorflow.keras.models import load_model, Model

from classes.sequences import SequenceBuilder, FreezePadding, FreezePadding_Non_Negative
from classes.metrics import specificity
from classes.io import read_data, read_dictionary
from classes.models import get_model_parameters, get_importances

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', 500)


def parse_arguments(parser):
    """Read user arguments"""
    parser.add_argument('--path_model',
                        type=str, default='Model/weights.006.hdf5',
                        help='Path to the model to evaluate')
    parser.add_argument('--path_data', type=str, default='data/data_test.pkl',
                        help='Path to evaluation data')
    parser.add_argument('--path_dictionary', type=str, default='data/dictionary.pkl',
                        help='Path to codes dictionary')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for initial probability predictions')
    # parser.add_argument('--id', type=int, default=0,
    #                     help='Id of the patient being interpreted')
    args = parser.parse_known_args()[0]
    return args


def main(ARGS):
    """Main Body of the code"""
    print('Loading Model and Extracting Parameters')
    model = load_model(ARGS.path_model, custom_objects={'specificity':specificity,
                                             'FreezePadding':FreezePadding,
                                             'FreezePadding_Non_Negative':FreezePadding_Non_Negative})
    #Interpret requires that we get the attention weights
    model_with_attention = Model(model.inputs, model.outputs +\
                                              [model.get_layer(name='softmax').output,\
                                               model.get_layer(name='beta_dense_0').output])
    model_parameters = get_model_parameters(model)

    print('Reading Data')
    data = read_data(ARGS, model_parameters, mode='eval')
    dictionary = read_dictionary(ARGS)

    print('Predicting and interpreting')
    pred_generator = SequenceBuilder(data, ARGS, target=None, model_parameters=model_parameters)
    probabilities = model.predict_generator(generator=pred_generator, verbose=2)

    ARGS.batch_size = 1
    while 1:
        patient_id = int(input('Input Patient Order Number: '))
        if patient_id > len(data[0]) - 1:
            print(f'Invalid ID, there are only {len(data[0])} patients')
        elif patient_id < 0:
            print('Only Positive IDs are accepted')
        else:
            print(f'Patients probability: {probabilities[patient_id, 0, 0]}')
            proceed = str(input('Output predictions? (y/n): '))
            if proceed == 'y':
                patient_data = pred_generator.__getitem__(patient_id)
                proba, alphas, betas = model_with_attention.predict_on_batch(patient_data)
                visits = get_importances(alphas[0], betas[0], patient_data,
                                         model_parameters, dictionary,
                                         path_icdpath='data/CMS32_DESC_LONG_DX.txt', code_prefix='D_')
                for visit in visits:
                    print(visit)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ARGS = parse_arguments(PARSER)
    main(ARGS)
