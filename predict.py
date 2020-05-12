import csv
import numpy as np
import random
import tokenizer as tk
from keras.models import load_model

class Predict:
    def __init__(self, k, dict_file_path, model_path):
        '''
        initialize the RNN model with the input dimension and length
        Args:
            k: the kmer value
            dict_file_path: file path to the dictionary
            model_path: the path to the model
        '''
        self.kmer = k
        self.model = load_model(model_path)

        mapping = list()
        # import the .csv as a list
        with open(dict_file_path, 'r') as readfile:
            seq_reader = csv.reader(readfile)
            for row in seq_reader:
                mapping.append(row[0])
        self.dict = mapping

    def predict(self, primary_seq):
        secondary_seq = []
        tokenized = []
        output = ''

        # tokenize the input sequence based on the input k value and dictionary
        for i in range(1, len(primary_seq), self.kmer):
            piece = primary_seq[i:i+self.kmer]
            try:
                tokenized.append(self.dict.index(piece))
            except:
                print(piece, 'not found ')
                tokenized.append(0)

        # format the features
        tokenized = np.asarray(tokenized)
        tokenized = np.reshape(tokenized, (len(tokenized), 1))

        # predict
        default = 'N'
        for k in range(1, self.kmer, 1):
            default = default + 'N'

        for t in range(len(tokenized)):
            if tokenized[t] != 0:
                prediction = (np.argmax(self.model.predict(tokenized[t])))
                secondary_seq.append(self.dict[prediction])
            else:
                secondary_seq.append(default)

        # stitch the list into a single string
        output = output.join(secondary_seq)

        return output


if __name__ == "__main__":
    lines = tk.purge('RNA_sequence_input.csv')
    random.shuffle(lines)
    sample = lines[1]

    print('input primary sequence')
    print(sample[0])
    print('expected secondary structure:')
    print(sample[1])
    print('\n')

    rnn_predictor = Predict(6, '6_kmer_RNN.csv', '6_kmer_RNN.h5')
    cnn_predictor = Predict(6, '6_kmer_CNN.csv', '6_kmer_CNN.h5')
    rnn_predicted = rnn_predictor.predict(sample[0])
    cnn_predicted = cnn_predictor.predict(sample[0])

    print('RNN prediction:')
    print(rnn_predicted)
    print('CNN prediction:')
    print(cnn_predicted)
