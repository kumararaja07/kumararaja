import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.preprocessing.sequence import pad_sequences

# Sample data
input_texts = ['I love NLP', 'He plays football']
target_texts = [['PRON', 'VERB', 'NOUN'], ['PRON', 'VERB', 'NOUN']]

# Vocabulary creation
word_vocab = sorted(set(word for sent in input_texts for word in sent.split()))
tag_vocab = sorted(set(tag for tags in target_texts for tag in tags))
word2idx = {word: i+1 for i, word in enumerate(word_vocab)}  # +1 for padding
tag2idx = {tag: i for i, tag in enumerate(tag_vocab)}
idx2tag = {i: tag for tag, i in tag2idx.items()}

# Parameters
max_len = max(len(sent.split()) for sent in input_texts)
vocab_size = len(word2idx) + 1
tag_size = len(tag2idx)

# Data preparation
encoder_input_data = pad_sequences([[word2idx[word] for word in sent.split()] for sent in input_texts], maxlen=max_len, padding='post')
decoder_output_data = pad_sequences([[tag2idx[tag] for tag in tags] for tags in target_texts], maxlen=max_len, padding='post')
decoder_output_data = np.expand_dims(decoder_output_data, -1)  # shape: (batch, seq_len, 1)

# Model definition
encoder_inputs = Input(shape=(max_len,))
x = Embedding(input_dim=vocab_size, output_dim=64)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(64, return_state=True)(x)

decoder_inputs = Input(shape=(max_len,))
y = Embedding(input_dim=vocab_size, output_dim=64)(decoder_inputs)
decoder_lstm = LSTM(64, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(y, initial_state=[state_h, state_c])
decoder_dense = Dense(tag_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit([encoder_input_data, encoder_input_data], decoder_output_data, epochs=100, verbose=0)

# Test case
test_sentence = "He loves NLP"
test_seq = [word2idx.get(word, 0) for word in test_sentence.split()]
test_seq_padded = pad_sequences([test_seq], maxlen=max_len, padding='post')
decoder_input_test = test_seq_padded

predictions = model.predict([test_seq_padded, decoder_input_test])
predicted_tags = [idx2tag[np.argmax(p)] for p in predictions[0]]
print("Input Sentence:", test_sentence)
print("Predicted POS tags:", predicted_tags)

Output:

<img width="869" height="148" alt="Screenshot 2025-09-17 095824" src="https://github.com/user-attachments/assets/33378a15-5db4-48cf-8150-4009db177583" />
