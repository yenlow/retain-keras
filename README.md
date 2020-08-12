# RETAIN-Keras: Keras reimplementation of RETAIN

This repository is an update of [retain-keras](https://github.com/Optum/retain-keras). It was reimplemented by Optum to use tf.keras 1.12. Here, it is updated to use with tf.keras 1.15

[RETAIN](https://github.com/mp2893/retain) is a neural network architecture originally introduced by [Choi et al.](https://arxiv.org/abs/1608.05745) to represent medical claims codes as embeddings and also predict diagnosis. It uses 2 Recurrent Neural Network models with double attentions weights. 

With these double attention weights to account for selected visits and codes, the resulting architecture is highly interpretable.
[![RETAIN Interpretation Demo](http://mp2893.com/images/thumbnail.png)](https://youtu.be/co3lTOSgFlA?t=1m46s "RETAIN Interpretation Demo by Choi - Click to Watch!")

## Improvements and Extra Features
* Renamed parameters to be more self-descriptive
* Set parameters to match Choi's paper
* Added ICD9 lookup to aid interpretation
* Updated with keras builtin callbacks (ModelCheckpoint, EarlyStopping, CSVLogger)
* Bypassed [bug](https://github.com/tensorflow/tensorflow/issues/33755) concerning embedding constraint
* Refactored code for training, evaluation, interpretation to re-use common classes and functions
* Disabled multi-gpu

Previous improvements by Optum
* Simpler Keras code with Tensorflow backend (tested for tf 1.12)
* Ability to use extra numeric inputs of fixed size that can hold numeric information about the patients visit such as patient's age, quantity of drug prescribed, or blood pressure
* Improved embedding logic that avoids using large dense inputs
* Ability to create multi-gpu models (experimental)
* Switch to CuDNN implementations of RNN layers that provides immense training speed ups
* Ability to evaluate models during training
* Ability to train models with only positive contributions which improves performance
* Extra script to evaluate the model and output several helper graphics

---
## Installation
Clone the repository
`git clone git@github.com:yenlow/retain-keras.git`

#### Requirements
1. Python 3 (tested with 3.7)
2. tensorflow (tested with 1.15)
3. Keras (2.1.3+)
4. Scikit-Learn
5. Numpy
6. Pandas
7. Matplotlib (evaluation)
8. If using GPU: CUDA and CuDNN

---
## Usage
#### 1. Data processing
Reshape to nested sequence lists, split data into training and test sets

`python process_mimic_modified.py ADMISSIONS.csv DIAGNOSES_ICD.csv PATIENTS.csv data .7`

#### 2. Training
Checkpoint models and log callback history

`python retain_train.py --additional arguments`
#### 3. Evaluation
Compute evaluation metrics on test set and save plots

`python retain_evaluation.py --additional arguments`
#### 4. Interpretation
Get feature/visit importance from attention weights

`python retain_interpretations.py --additional arguments`

### Run parameters
##### Training Arguments
retain_train.py has multiple arguments to customize the training and model:
* `--num_codes`: Integer number of medical codes in the data set. Think vocabulary size.
* `--numeric_size`: Integer number of numeric input features, 0 if none. Default: 0
* `--use_time`: Whether to use time as input for each visit. Default: False
* `--emb_size`: Integer imension of the visit embedding. Default: 256
* `--epochs`: Integer Number of epochs for training. Default: 30
* `--max_seq_len`: Maximum number of visits after which the data is truncated (think max sequence length for padding). This features helps to conserve GPU Ram (only the most recent max_seq_len will be used). Default: 300
* `--recurrent_size`: Integer dimension of hidden recurrent layers. Default: 256
* `--path_data_train`: String path to train data. Default: 'data/data_train.pkl'
* `--path_data_test`: String path to test/validation data. Default: 'data/data_test.pkl'
* `--path_target_train`: String path to train target. Default: 'data/target_train.pkl'
* `--path_target_test`: String path to test/validation target. Default: 'data/target_test.pkl'
* `--batch_size`: Integer batch size for training. Default: 100
* `--dropout_input`: Float dropout rate for embedding of codes and numeric features (0 to 1). Default: 0.4
* `--dropout_context`: Float dropout rate for context vector (0 to 1). Default: 0.6
* `--l2`: Float L2 regularization value for layers. Default: 0.0001
* `--out_directory`: String output directory to save the model, callback logs and evaluation_plots. Default: 'Model' 
* `--allow_negative`: Allows negative weights for embeddings/attentions (original RETAIN implementation allows it but forcing non-negative weights have shown to perform better on a range of tasks). Default: False

##### Evaluation Arguments
retain_evaluation.py has some arguments:
* `--path_model`: String path to the model to evaluate. Default: 'Model/weights.01.hdf5'
* `--path_data`: String path to evaluation data. Default: 'data/data_test.pkl'
* `--path_target`: String path to evaluation target. Default: 'data/target_test.pkl'
* `--graphs`: Whether to plot and save evaluation graphs. Default: False (i.e. no graphs)
* `--max_seq_len`: Integer maximum number of visits after which the data is truncated (think max sequence length for padding). Default: 300
* `--batch_size`: Integer batch size for prediction (higher values are generally faster). Default: 100
* `--out_directory`: String output directory to save the model, callback logs and evaluation png. Default: 'Model' 

##### Interpretation Arguments
retain_interpretations.py has some arguments:
* `--path_model`: String path to the model to evaluate. Default: 'Model/weights.01.hdf5'
* `--path_data`: String path to evaluation data. Default: 'data/data_test.pkl'
* `--path_dictionary`: Path to dictionary pkl that maps claim code to a alphanumeric key. If numerics inputs are used, they should have indexes num_codes+1 through num_codes+numeric_size, num_codes index is reserved for padding. Default:'data/dictionary.pkl'
* `--batch_size`: Integer batch size for prediction (higher values are generally faster). Default: 32

---
### Inputs format
#### Data
By default data has to be saved as a pickled pandas dataframe with following format:
* Each row is 1 patient
* Rows are sorted by the number of visits a person has. People with the least visits should be in the beginning of the dataframe and people with the most visits at the end
* Column 'codes' is a list of lists where each sublist are codes for the individual visit. Lists have to ordered by their order of events (from old to new)
* Column 'numerics' is a list of lists where each sublist contains numeric values for  individual visit. Lists have to be ordered by their order of events (from old to new). Lists have to have a static size of numeric_size indicating number of  different numeric features for each visit. Numeric information can include things like patients age, blood pressure, BMI, length of the visit, or cost charged (or all at the same time!). This column is not used if numeric_size equals 0
* Column 'to_event' is a list of values indicating when the respective visit happened. Values have to be ordered from oldest to newest. This column is not used if use_time is not specified

#### Target
By default target has to be saved as a pickled pandas dataframe with following format:
* Each row is 1 patient corresponding to the patient from data file
* Column 'target' is patient's class (either 0 or 1)

---
## Sample Data Generation Using MIMIC-III
You can quickly test this reimplementation by creating a sample dataset from MIMIC-III data using process_mimic_modified.py script

You will need to request access to [MIMIC-III](https://mimic.physionet.org/gettingstarted/access/), a de-identified database containing information about clinical care of patients for 11 years of data, to be able to run this script.
Or, you can download the [MIMIC-III](https://physionet.org/content/mimiciii-demo/1.4/) sample demo data (~100 patients only) and use it for exploratory benchmarks. 

This script heavily borrows from original [process_mimic.py](https://github.com/mp2893/retain/blob/master/process_mimic.py) created by Edward Choi but is modified to output data in a format specified above. It outputs the necessary files to a user-specified directory and splits them into train and test by user-specified ratio.

Example:

Run from the MIMIC-III directory. This will split data with 70% going to training and 30% to test:  

`python process_mimic_modified.py ADMISSIONS.csv DIAGNOSES_ICD.csv PATIENTS.csv data .7`

---
## Licenses and Contributions
Please review the [license](LICENSE), [notice](Notice.txt) and other [documents](docs/) before using the code in this repository or making a contribution to the repository

## References
	RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism
	Edward Choi, Mohammad Taha Bahadori, Joshua A. Kulas, Andy Schuetz, Walter F. Stewart, Jimeng Sun,
	NIPS 2016, pp.3504-3512
	
	Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. 
	PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. 
	Circulation 101(23):e215-e220 [Circulation Electronic Pages; 
	http://circ.ahajournals.org/content/101/23/e215.full]; 2000 (June 13).
