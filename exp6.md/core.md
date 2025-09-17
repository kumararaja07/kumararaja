import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed
from keras.preprocessing.sequence import pad_sequences

# Sample data
input_texts = ['I love NLP', 'He plays football']
target_texts = [['PRON', 'VERB', 'NOUN'], ['PRON', 'VERB', 'NOUN']]

# Vocabulary
word_vocab = sorted(set(word for sent in input_texts for word in sent.split()))
tag_vocab = sorted(set(tag for tags in target_texts for tag in tags))
word2idx = {word: i+1 for i, word in enumerate(word_vocab)}  # +1 for padding
tag2idx = {tag: i for i, tag in enumerate(tag_vocab)}
idx2tag = {i: tag for tag, i in tag2idx.items()}

# Parameters
max_len = max(len(sent.split()) for sent in input_texts)
vocab_size = len(word2idx) + 1
tag_size = len(tag2idx)

# Prepare data
X = pad_sequences([[word2idx[word] for word in sent.split()] for sent in input_texts], maxlen=max_len, padding='post')
y = pad_sequences([[tag2idx[tag] for tag in tags] for tags in target_texts], maxlen=max_len, padding='post')
y = np.expand_dims(y, -1)

# Model
input_layer = Input(shape=(max_len,))
embedding = Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len)(input_layer)
lstm = LSTM(64, return_sequences=True)(embedding)
output = TimeDistributed(Dense(tag_size, activation='softmax'))(lstm)

model = Model(input_layer, output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=0)

# Test
test_sentence = "He loves NLP"
test_seq = [word2idx.get(word, 0) for word in test_sentence.split()]
test_seq_padded = pad_sequences([test_seq], maxlen=max_len, padding='post')
pred = model.predict(test_seq_padded)
pred_tags = [idx2tag[np.argmax(p)] for p in pred[0]]
print("Input:", test_sentence)
print("Predicted POS tags:", pred_tags)<img width="1278" height="101" alt="image" src="https://github.com/user-attachments/assets/903c2ba9-46c6-4936-b991-9074ad9adf92" />
