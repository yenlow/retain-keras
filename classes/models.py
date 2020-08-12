import re
import numpy as np
import pandas as pd
from classes.icd import read_icd9dict

class ModelParameters:
    """Helper class to store model parameters in the same format as ARGS"""
    def __init__(self):
        self.num_codes = None
        self.numeric_size = None
        self.use_time = None
        self.emb_weights = None
        self.output_weights = None
        self.bias = None


def get_model_parameters(model):
    """Extract model arguments that were used during training"""
    params = ModelParameters()
    names = [layer.name for layer in model.layers]

    params.num_codes = model.get_layer(name='embedding').input_dim-1
    params.emb_weights = model.get_layer(name='embedding').get_weights()[0]
    params.output_weights, params.bias = model.get_layer(name='time_distributed_out').get_weights()

    print('Model bias: {}'.format(params.bias))

    if 'numeric_input' in names:
        params.numeric_size = model.get_layer(name='numeric_input').input_shape[2]

        #Add artificial embeddings for each numeric feature and extend the embedding weights
        #Numeric embeddings is just 1 for 1 dimension of the embedding which corresponds to taking value as is
        numeric_embeddings = np.zeros((params.numeric_size, params.emb_weights.shape[1]+params.numeric_size))
        for i in range(params.numeric_size):
            numeric_embeddings[i, params.emb_weights.shape[1]+i] = 1
        #Extended embedding is original embedding extended to larger output size and numerics embeddings added
        params.emb_weights = np.append(params.emb_weights,
                                       np.zeros((params.num_codes+1, params.numeric_size)),
                                       axis=1)
        params.emb_weights = np.append(params.emb_weights, numeric_embeddings, axis=0)

    else: #no 'numeric input' in layer names
        params.numeric_size = 0
    if 'time_input' in names:
        params.use_time = True
    else: #no 'time input' in layer names
        params.use_time = False

    return params


def get_importances(alphas, betas, patient_data, model_parameters, dictionary, path_icdpath='data/CMS32_DESC_LONG_DX.txt', code_prefix='D_'):
    """Construct dataframes that interpret each visit of the given patient"""
    icd9dict = read_icd9dict(file=path_icdpath, widths=[5, 10000])

    importances = []
    codes = patient_data[0][0]
    index = 1
    if model_parameters.numeric_size:
        numerics = patient_data[index][0]
        index += 1

    if model_parameters.use_time:
        time = patient_data[index][0].reshape((len(codes),))
    else:
        time = np.arange(len(codes))

    for i in range(len(patient_data[0][0])):
        visit_codes = codes[i]
        visit_beta = betas[i]
        visit_alpha = alphas[i][0]
        relevant_indices = np.append(visit_codes,
                                     range(model_parameters.num_codes+1,
                                           model_parameters.num_codes+1+model_parameters.numeric_size))\
                                          .astype(np.int32)
        values = np.full(fill_value='Diagnosed', shape=(len(visit_codes),))
        if model_parameters.numeric_size:
            visit_numerics = numerics[i]
            values = np.append(values, visit_numerics)
        values_mask = np.array([1. if value == 'Diagnosed' else value for value in values], dtype=np.float32)
        beta_scaled = visit_beta * model_parameters.emb_weights[relevant_indices]
        output_scaled = np.dot(beta_scaled, model_parameters.output_weights)
        alpha_scaled = values_mask * visit_alpha * output_scaled

        features = []
        desc = []
        for index in relevant_indices:
            f = dictionary[index]
            features.append(f)
            desc.append(icd9dict.get(re.sub(rf"^{code_prefix}",'',f,count=1)))
        df_visit = pd.DataFrame({'outcome': values,
                                 'feature': features,
                                 'description': desc,
                                 'impt_of_feature': alpha_scaled[:, 0],
                                 'impt_of_visit': visit_alpha,
                                 'to_event': time[i]},
                                columns=['outcome', 'feature', 'description',
                                         'impt_of_feature','impt_of_visit', 'to_event'])
        df_visit = df_visit[df_visit['feature'] != dictionary[len(dictionary)]] #padding
        df_visit.sort_values(['impt_of_feature'], ascending=False, inplace=True)
        importances.append(df_visit)

    return importances
