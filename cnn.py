from keras.models import Sequential
from keras.layers import Dense, Dropout, Masking, Embedding, Conv1D, MaxPool1D, GlobalMaxPool1D
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

import draft as d

class CNN:
    def __init__(self, dim, length, modelname='cnn_model'):
        self.model = Sequential()

        self.model.add(Embedding(input_dim=dim,
                            input_length=length,
                            output_dim=100,
                            trainable=True,
                            mask_zero=False))
        
        self.model.add(Conv1D(64, 3, padding='same'))
        self.model.add(MaxPool1D(pool_size=2, padding='same'))
        self.model.add(Conv1D(32, 3, padding='same'))
        self.model.add(MaxPool1D(pool_size=2, padding='same'))
        self.model.add((Conv1D(16, 3, padding='same')))
        self.model.add(GlobalMaxPool1D())
        # self.model.add(Dropout(0.2))
        self.model.add(Dense(100, activation='sigmoid'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(dim, activation='softmax'))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.callbacks = [ModelCheckpoint('saved_models/'+modelname+'.h5', save_best_only=True, save_weights_only=False)]
    
    def train(self, x_train, y_train, x_test, y_test, e=20, batch_size=256):
        history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=e, callbacks=self.callbacks, validation_data=(x_test, y_test))

        return history

def plot_vals(history, modelname):
    fig, ax = plt.subplots()
    # history.loc[:, 'val_accuracy'].plot(x='Epoch', y='Validation Accuracy', ax=ax)
    plt.plot(history.loc[:, 'val_accuracy'])
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.savefig('plots/{}_val_accuracy.png'.format(modelname))

def manage_runs(e=10, sample_rate=0.3):
    # General Parameters
    batch_size = 256

    # General Data Processing
    lines = d.purge('purged_RNA_secondary_structure.csv')

    # Run model
    for k in range(4, 11, 2):
        train_model(lines, k, batch_size, e, sample_rate=sample_rate)

def train_model(lines, k, batch_size, epochs, sample_rate):
    chosen_inds = np.random.choice(lines.shape[0], int(lines.shape[0] * 0.4))
    lines = lines[chosen_inds]
    random.shuffle(lines)

    tokenized_sequence, indexed_word = d.sequence_tokenizer(lines, k)
    vocab_len = len(indexed_word) + 1

    features, labels = d.feature_label_extractor(tokenized_sequence, vocab_len)

    # Train/Test Split
    split = int(0.8 * len(features))
    x_train = features[:split]
    x_test = features[split:]
    y_train = labels[:split]
    y_test = labels[split:]

    # Train Model
    modelname = 'cnn_{}'.format(k)
    model = CNN(vocab_len, 1, modelname)
    print(model.model.summary())
    history = model.train(x_train, y_train, x_test, y_test, batch_size=batch_size, e=epochs)

    # Save history and figures
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('model_history/'+modelname+'_data.csv')
    plot_vals(history_df, modelname)

if __name__ == '__main__':
    manage_runs(e=10)

    # # Parameters
    # k = 4
    # batch_size = 256
    # epochs = 2 

    # # Data Preprocessing
    # lines = np.array(d.purge('purged_RNA_secondary_structure.csv'))
    # chosen_inds = np.random.choice(lines.shape[0], int(lines.shape[0] * 0.4))
    # lines = lines[chosen_inds]
    # random.shuffle(lines)

    # tokenized_sequence, indexed_word = d.sequence_tokenizer(lines, k)
    # vocab_len = len(indexed_word) + 1

    # features, labels = d.feature_label_extractor(tokenized_sequence, vocab_len)

    # # Train/Test Split
    # split = int(0.8 * len(features))
    # x_train = features[:split]
    # x_test = features[split:]
    # y_train = labels[:split]
    # y_test = labels[split:]

    # # Train Model
    # modelname = 'cnn_{}'.format(k)
    # model = CNN(vocab_len, 1, modelname)
    # print(vocab_len)
    # print(model.model.summary())
    # history = model.train(x_train, y_train, x_test, y_test, batch_size=batch_size, e=epochs)
    # print(history)
    # history_df = pd.DataFrame(history.history)
    # history_df.to_csv('model_history/'+modelname+'_data.csv')
    # plot_vals(history_df, modelname)
