import csv
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import random


def purge(filepath):
    """
    The purge method will extract data from the input .csv file and
    remove rows that have <100 RNA nucleotides
    Args:
        filepath: the file path to the .csv file

    Returns:
        a list that contains the RNA sequence and secondary structure arrays
    """
    lines = list()
    # import the .csv as a list
    with open(filepath, 'r') as readfile:
        seq_reader = csv.reader(readfile)
        for row in seq_reader:
            if len(row[1]) > 2600:
                lines.append([row[1], row[2]])

    print(len(lines), " lines saved after purging ")
    return lines


def sequence_tokenizer(lines, kmer):
    """
    The sequence_tokenizer will tokenize the sequence and convert the token to
    numeric values for training
    Args:
        lines: the input list that contains arrays of RNA sequence and secondary structures
        kmer: the k-mer length for parsing the sequences

    Returns:
        sequences: the tokenized sequence in numeric representation
        idx_word: a set that has the number to sequence mapping
    """
    parsed_list = list()
    # for i in range(len(lines)):
    #     sequence = lines[i][0]
    #     structure = lines[i][1]
    #     sub_parsed_list = list()
    #     group_start = 0
    #
    #     for j in range(len(structure)):
    #         if structure[j] != structure[group_start]:
    #             if structure[group_start] == '(':
    #                 sub_parsed_list.append([sequence[group_start: j], '('])
    #             elif structure[group_start] == ')':
    #                 sub_parsed_list.append([sequence[group_start: j], ')'])
    #             elif structure[group_start] == '<':
    #                 sub_parsed_list.append([sequence[group_start: j], '<'])
    #             elif structure[group_start] == '>':
    #                 sub_parsed_list.append([sequence[group_start: j], '>'])
    #             else:
    #                 sub_parsed_list.append([sequence[group_start: j], '.'])
    #             group_start = j
    #     parsed_list.extend(sub_parsed_list)

    for i in range(len(lines)):
        sequence = lines[i][0]
        structure = lines[i][1]
        sub_parsed_list = list()

        for j in range(len(structure) - kmer):
            sub_parsed_list.append([sequence[j: j + kmer], structure[j: j + kmer]])
        parsed_list.extend(sub_parsed_list)

    token = Tokenizer(num_words=None, filters='', lower=False, split=' ')
    # train the tokenizer and convert nucleotide into integers
    token.fit_on_texts(parsed_list)
    sequences = token.texts_to_sequences(parsed_list)
    # save the mapping
    idx_word = token.index_word

    print(len(sequences), " sequences are tokenized")
    return sequences, idx_word


def feature_label_extractor(sequences):
    """
    The function will generate the feature and label arrays
    Args:
        sequences: the input sequences

    Returns:
        features: the features array
        label_array: one-hot coded label array
    """
    features = []
    labels = []

    for i in range(len(sequences)):
        features.append(sequences[i][0])
        labels.append(sequences[i][1])

    features = np.asarray(features)
    features = np.reshape(features, (len(features), 1))

    # one-hot code the labels
    label_array = np.zeros((len(features), vocab_len), dtype = np.int8)
    for example_index, word_index in enumerate(labels):
        label_array[example_index, word_index] = 1

    print("created features with a dim of ", features.shape)
    print("created one-hot coded labels with a dim of ", label_array.shape)
    return features, label_array


class RNN:
    def __init__(self, dim, length):
        """
        initialize the RNN model with the input dimension and length
        Args:
            dim: the dimension is the length of the unique vocabulary + 1
            length: The column number for the input feature array
        """
        self.model = Sequential()
        # Embedding layer
        self.model.add(Embedding(input_dim=dim,
                            input_length=length,
                            output_dim=100,
                            trainable=True,
                            mask_zero=True))
        # Recurrent layer
        self.model.add(LSTM(64,
                       return_sequences=False,
                       dropout=0.1,
                       recurrent_dropout=0.1))
        # Fully connected layer
        self.model.add(Dense(64, activation='relu'))
        # Dropout for regularization
        self.model.add(Dropout(0.5))
        # Output layer
        self.model.add(Dense(dim, activation='softmax'))
        # Compile the model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # create callback
        self.call_backs = [EarlyStopping(monitor='val_loss', patience=5),
                           ModelCheckpoint('model.h5', save_best_only=True, save_weights_only=False)]
        print("model is initialized with ", dim, " input_dim and ", length, " input length")

    def train(self, x_train, y_train, x_test, y_test):
        print("training the model with ", len(x_train), " training data")
        history = self.model.fit(x_train,
                                 y_train,
                                 batch_size=256,
                                 epochs=20,
                                 callbacks=self.call_backs,
                                 validation_data=(x_test, y_test))

        print("completed training")
        return history


if __name__ == "__main__":
    kmer = 4

    # get a list of RNA sequence and secondary structures
    lines = purge('purged_RNA_secondary_structure.csv')

    # shuffle the data
    random.shuffle(lines)

    # run the tokenizer
    tokenized_sequence, indexed_word = sequence_tokenizer(lines, kmer)
    vocab_len = len(indexed_word) + 1

    # get feature and labels
    features, labels = feature_label_extractor(tokenized_sequence)

    # split training and test sets
    split = int(0.8 * len(features))
    x_train = features[:split]
    x_test = features[split:]
    y_train = labels[:split]
    y_test = labels[split:]

    # initialize the RNN
    model = RNN(vocab_len, 1)

    # train
    history = model.train(x_train, y_train, x_test, y_test)
    print(history)




