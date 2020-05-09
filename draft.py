import random
from datetime import datetime
import tokenizer as tk
import rnn as rnn
import cnn as cnn


if __name__ == "__main__":
    kmer = 4
    print("training with", kmer, "kmer")

    # get a list of RNA sequence and secondary structures
    lines = tk.purge('purged_RNA_secondary_structure.csv')

    # shuffle the data
    random.shuffle(lines)

    # run the tokenizer
    tokenized_sequence, indexed_word = tk.sequence_tokenizer(lines, kmer, 1)
    vocab_len = len(indexed_word) + 1

    # get feature and labels
    features, labels = tk.feature_label_extractor(tokenized_sequence, vocab_len)

    # split training and test sets
    split = int(0.8 * len(features))
    x_train = features[:split]
    x_test = features[split:]
    y_train = labels[:split]
    y_test = labels[split:]

    # initialize the RNN
    model = rnn.RNN(vocab_len, 1)

    # train
    history = model.train(x_train, y_train, x_test, y_test)
    now = datetime.now()
    end_time = now.strftime("%b-%d-%Y %H:%M:%S")
    print("training completed", end_time)

    # # Load in model and evaluate on validation data
    # model = load_model('May-03-2020_13-38-01.h5')
    # model.summary()
    # results = model.evaluate(x_test, y_test)
    # print('test loss, test accuracy:', results)
    #
    # # try out a prediction
    # print("RNA sequence is:\n", x_test[:1])
    # predictions = model.predict(x_test[:1])
    # print('prediction:\n', predictions)
    # print('actual:\n', y_test[:1])



