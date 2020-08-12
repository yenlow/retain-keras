"""RETAIN Model Evaluation"""
import argparse
from tensorflow.keras.models import load_model

from classes.metrics import specificity, precision_recall, probability_calibration, lift, roc
from classes.sequences import SequenceBuilder, FreezePadding, FreezePadding_Non_Negative
from classes.io import read_data
from classes.models import get_model_parameters


def parse_arguments(parser):
    """Read user arguments"""
    parser.add_argument('--path_model',
                        type=str, default='Model/weights.006.hdf5',
                        help='Path to the model to evaluate')
    parser.add_argument('--path_data', type=str, default='data/data_test.pkl',
                        help='Path to evaluation data')
    parser.add_argument('--path_target', type=str, default='data/target_test.pkl',
                        help='Path to evaluation target')
    parser.add_argument('--graphs', action='store_true',
                        help='Does not output graphs if argument is present')
    parser.add_argument('--max_seq_len', type=int, default=300,
                        help='Maximum number of visits after which the data is truncated')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for prediction (higher values are generally faster)')
    parser.add_argument('--out_directory', type=str, default='Model',
                        help='out_directory to save model(s), callback logs and evaluation pngs')
    args = parser.parse_known_args()[0]
    return args


def main(ARGS):
    """Main Body of the code"""

    print('Loading Model and Extracting Parameters')
    model = load_model(ARGS.path_model, custom_objects={'specificity': specificity,
                                             'FreezePadding':FreezePadding,
                                             'FreezePadding_Non_Negative':FreezePadding_Non_Negative})
    model_parameters = get_model_parameters(model)

    print('Reading Data')
    data, y = read_data(ARGS, model_parameters, 'eval')

    print('Predicting the probabilities')
    pred_generator = SequenceBuilder(data, ARGS, target=None, model_parameters=model_parameters)
    probabilities = model.predict_generator(generator=pred_generator, verbose=2)

    print('Evaluating')
    roc(y, probabilities[:, 0, -1], ARGS)
    precision_recall(y, probabilities[:, 0, -1], ARGS)
    lift(y, probabilities[:, 0, -1], ARGS)
    probability_calibration(y, probabilities[:, 0, -1], ARGS)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ARGS = parse_arguments(PARSER)
    main(ARGS)
