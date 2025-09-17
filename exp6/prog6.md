import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# Sample data
input_texts = ['I love NLP', 'He plays football']
target_texts = [['PRON', 'VERB', 'NOUN'], ['PRON', 'VERB', 'NOUN']]

# Vocabulary setup
word_vocab = sorted(set(word for sent in input_texts for word in sent.split()))
tag_vocab = sorted(set(tag for tags in target_texts for tag in tags))

word2idx = {word: i+1 for i, word in enumerate(word_vocab)}  # +1 for padding
tag2idx = {tag: i for i, tag in enumerate(tag_vocab)}
idx2tag = {i: tag for tag, i in tag2idx.items()}

# Parameters
max_len = max(len(sent.split()) for sent in input_texts)
num_words = len(word2idx) + 1  # +1 for padding
num_tags = len(tag2idx)

# Prepare encoder input
encoder_input_data = [[word2idx[word] for word in sent.split()] for sent in input_texts]
encoder_input_data = pad_sequences(encoder_input_data, maxlen=max_len, padding='post')

# Prepare decoder output
decoder_output_data = [[tag2idx[tag] for tag in tags] for tags in target_texts]
decoder_output_data = pad_sequences(decoder_output_data, maxlen=max_len, padding='post')
decoder_output_data = to_categorical(decoder_output_data, num_classes=num_tags)

# Model definition
encoder_inputs = Input(shape=(max_len,))
x = Embedding(input_dim=num_words, output_dim=64)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(64, return_state=True)(x)

decoder_inputs = Input(shape=(max_len,))
y = Embedding(input_dim=num_words, output_dim=64)(decoder_inputs)
decoder_lstm = LSTM(64, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(y, initial_state=[state_h, state_c])
decoder_dense = Dense(num_tags, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary
model.summary()
output:
<img width="780" height="593" alt="image" src="https://github.com/user-attachments/assets/edb38d2a-2fa6-42aa-9a2e-93b9bba93c72" />

