import random
import tokenizer as tk
import rnn as rnn
import cnn as cnn
import post_process as pp


if __name__ == "__main__":
    kmer = 2
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

    # initialize the models
    rnn_model_name = str(kmer) + '_kmer_RNN'
    cnn_model_name = str(kmer) + '_kmer_CNN'
    rnn_model = rnn.RNN(vocab_len, 1, rnn_model_name)
    cnn_model = cnn.CNN(vocab_len, 1, cnn_model_name)

    # train
    rnn_history = rnn_model.train(x_train, y_train, x_test, y_test)
    cnn_history = cnn_model.train(x_train, y_train, x_test, y_test)
    pp.save_and_plot(rnn_history, rnn_model_name)
    pp.save_and_plot(cnn_history, cnn_model_name)

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



