from tensorflow.keras.callbacks import Callback
from sklearn.metrics import average_precision_score

from classes.sequences import SequenceBuilder

class LogEval(Callback):
    '''
    Custom callback that logs selected metrics to text file

    Example:
        >>> custom_callback = LogEval(filepath, model, data, ARGS, interval=1, extlog=True)
        >>> model.fit(X_train, Y_train, callbacks=[custom_callback])
    '''
    def __init__(self, filepath, model, data, ARGS, interval=1, extlog=True):
        super(Callback, self).__init__()
        self.filepath = filepath
        self.interval = interval
        self.data_test, self.y_test = data
        # generator for data_test
        self.generator = SequenceBuilder(data=self.data_test, target=self.y_test,
                                         ARGS=ARGS, target_out=False)
        self.model = model
        self.extlog = extlog

    def on_epoch_end(self, epoch, logs={}):
        '''
        Compute average precision for validation data every interval epochs
        Log to Model/log.txt as specified by self.filepath or ARGS.out_directory
        Args:
            epoch (int):
            logs (dict):

        Returns: None
        '''

        if self.extlog and epoch == 0:
            file_output = open(self.filepath, 'w')
            file_output.write("epoch\tAUPRC\n")

        if epoch % self.interval == 0:
            # Compute predictions of the model
            y_pred = [x[-1] for x in
                      self.model.predict_generator(self.generator,
                                                   verbose=0,
                                                   use_multiprocessing=False)]
            auprc = average_precision_score(self.y_test, y_pred)

            result_string = "{:d}\t{:.5f}\n".format(epoch, auprc)
            print(result_string)

            # Create log files if it doesn't exist, otherwise write to it
            # Model/log.txt as specified by ARGS.out_directory
            if self.extlog:
                file_output = open(self.filepath, 'a')
                file_output.write(result_string)
