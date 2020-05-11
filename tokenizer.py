import csv
import numpy as np
from keras.preprocessing.text import Tokenizer


def purge(filepath):
    '''
    The purge method will extract data from the input .csv file and
    remove rows that have <100 RNA nucleotides
    Args:
        filepath: the file path to the .csv file

    Returns:
        a list that contains the RNA sequence and secondary structure arrays
    '''
    lines = list()
    # import the .csv as a list
    with open(filepath, 'r') as readfile:
        seq_reader = csv.reader(readfile)
        for row in seq_reader:
            if len(row[1]) > 2000:
                lines.append([row[1], row[2]])

    print(len(lines), " lines saved after purging ")
    return lines


def sequence_tokenizer(lines, kmer, increment, modelname):
    '''
    The sequence_tokenizer will tokenize the sequence and convert the token to
    numeric values for training
    Args:
        lines: the input list that contains arrays of RNA sequence and secondary structures
        kmer: the k-mer length for parsing the sequences
        increment: the step size for the iteration loop
        modelname: the model name

    Returns:
        sequences: the tokenized sequence in numeric representation
        idx_word: a set that has the number to sequence mapping
    '''
    parsed_list = list()

    if increment <= kmer:
        for i in range(len(lines)):
            sequence = lines[i][0]
            structure = lines[i][1]
            sub_parsed_list = list()

            for j in range(0, len(structure) - kmer, increment):
                sub_parsed_list.append([sequence[j: j + kmer], structure[j: j + kmer]])
            parsed_list.extend(sub_parsed_list)
    else:
        print("invalid input: increment size > kmer")

    token = Tokenizer(num_words=None, filters='', lower=False, split=' ')
    # train the tokenizer and convert nucleotide into integers
    token.fit_on_texts(parsed_list)
    sequences = token.texts_to_sequences(parsed_list)
    # save the mapping
    idx_word = token.index_word

    with open(modelname + '.csv', 'w') as file:
        fieldnames = ['symbols']
        writer = csv.DictWriter(file, fieldnames = fieldnames)
        writer.writeheader()
        for i in range(1, len(idx_word), 1):
            writer.writerow({'symbols':idx_word[i]})

    print(len(sequences), " sequences are tokenized")
    return sequences, idx_word


def feature_label_extractor(sequences, vocab_len):
    '''
    The function will generate the feature and label arrays
    Args:
        sequences: the input sequences
        vocab_len: the lenght of the vocabulary array

    Returns:
        features: the features array
        label_array: one-hot coded label array
    '''
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
