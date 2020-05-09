from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime


class RNN:
    def __init__(self, dim, length, model_name='rnn_model'):
        '''
        initialize the RNN model with the input dimension and length
        Args:
            dim: the dimension is the length of the unique vocabulary + 1
            length: The column number for the input feature array
        '''
        # datetime object containing current date and time
        now = datetime.now()
        format_date = now.strftime("%b-%d-%Y_")

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
                           ModelCheckpoint(model_name + '.h5',
                                           save_best_only=True,
                                           save_weights_only=False)]
        print("model is initialized with ", dim, " input_dim and ", length, " input length")

    def train(self, x_train, y_train, x_test, y_test):
        print("training the model with ", len(x_train), " training data")
        history = self.model.fit(x_train,
                                 y_train,
                                 batch_size=256,
                                 epochs=10,
                                 callbacks=self.call_backs,
                                 validation_data=(x_test, y_test))

        print("completed training")
        return history
