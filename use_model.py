from keras.preprocessing import sequence
import keras 
import tensorflow as tf
import os
import numpy as np

class shakes_ai:
    def __init__(self):
        self.path_to_file = keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
        self.text = open(self.path_to_file, 'rb').read().decode(encoding='utf-8')
        self.vocab = sorted(set(self.text))
        self.char2idx = {u:i for i, u in enumerate(self.vocab)}
        self.idx2char = np.array(self.vocab)
        self.BATCH_SIZE = 64
        self.VOCAB_SIZE = len(self.vocab)
        self.EMBEDDING_DIM = 256
        self.RNN_UNITS = 1024
        self.BUFFER_SIZE = 10000
        self.checkpoint_dir = './training_checkpoints'

    def build_model(self,vocab_size, embedding_dim, rnn_units, batch_size):
        model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size,embedding_dim,batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
        return model

    def generate_text(self,start_string,num_generate=2000):
        model = self.build_model(self.VOCAB_SIZE, self.EMBEDDING_DIM, self.RNN_UNITS, batch_size=1)
        model.load_weights(tf.train.latest_checkpoint(self.checkpoint_dir))
        model.build(tf.TensorShape([1, None]))
        #num_generate = 500
        input_eval = [self.char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        text_generated = []
        temprature = 1.0

        model.reset_states()
        for i in range(num_generate):
            predictions = model(input_eval)
            predictions = tf.squeeze(predictions, 0)

            predictions = predictions/temprature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

            input_eval = tf.expand_dims([predicted_id], 0)
            text_generated.append(self.idx2char[predicted_id])
        
        return (start_string + ''.join(text_generated))