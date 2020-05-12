import random
import tokenizer as tk
import rnn as rnn
import cnn as cnn
import post_process as pp


if __name__ == "__main__":
    epochs = 20

    # get a list of RNA sequence and secondary structures
    lines = tk.purge('RNA_sequence_input.csv')
#    lines = lines[:100]

    # Run All
    for kmer in range(4, 11, 2):
        print('Starting k={} Training:\n'.format(kmer))

        # RNN
        print('Training RNN\n')
        rnn_model_name = str(kmer) + '_kmer_RNN'
        # shuffle the data
        random.shuffle(lines)

        # run the tokenizer
        tokenized_sequence, indexed_word = tk.sequence_tokenizer(lines, kmer, 1, rnn_model_name)
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
        rnn_model = rnn.RNN(vocab_len, 1, rnn_model_name)

        # train and save outputs
        rnn_history = rnn_model.train(x_train, y_train, x_test, y_test, e=epochs)
        pp.save_and_plot(rnn_history, rnn_model_name)
        print('Trained RNN saved, outputs generated and saved.')

        # CNN
        print('Training CNN')
        cnn_model_name = str(kmer) + '_kmer_CNN'
        # shuffle the data
        random.shuffle(lines)

        # run the tokenizer
        tokenized_sequence, indexed_word = tk.sequence_tokenizer(lines, kmer, 1, cnn_model_name)
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
        cnn_model = cnn.CNN(vocab_len, 1, cnn_model_name)

        # train
        cnn_history = cnn_model.train(x_train, y_train, x_test, y_test, e=epochs)
        pp.save_and_plot(cnn_history, cnn_model_name)
        print('Trained CNN saved, outputs generated and saved.')

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



