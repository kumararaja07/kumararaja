
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Step 1: Sample reviews and labels
reviews = [
    "An emotional and deep plot",
    "The story was dull",
    "Absolutely loved the characters",
    "Terrible and boring movie",
    "It was okay, not great"
]

labels = np.array([1, 0, 1, 0, 0])  # 1 = Positive, 0 = Negative

# Step 2: Tokenize and pad
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
X = pad_sequences(sequences, maxlen=20)

# Step 3: Split data
X_train, X_test = X[:3], X[3:]
y_train, y_test = labels[:3], labels[3:]

# Step 4: Build LSTM model
lstm_model = Sequential([
    Embedding(input_dim=10000, output_dim=32, input_length=20),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
lstm_model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=0)

# Step 5: Build GRU model
gru_model = Sequential([
    Embedding(input_dim=10000, output_dim=32, input_length=20),
    GRU(64, dropout=0.2, recurrent_dropout=0.2),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
gru_model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
gru_model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=0)

# Step 6: Predict on test data
lstm_preds = (lstm_model.predict(X_test) > 0.5).astype(int).flatten()
gru_preds = (gru_model.predict(X_test) > 0.5).astype(int).flatten()

# Step 7: Display comparison table
print(f"{'Review Text':<35} {'Expected':<10} {'LSTM Output':<13} {'GRU Output':<12} {'Same?'}")
print("-" * 85)
for review, expected, lstm, gru in zip(reviews[3:], y_test, lstm_preds, gru_preds):
    expected_label = "Positive" if expected == 1 else "Negative"
    lstm_label = "Positive" if lstm == 1 else "Negative"
    gru_label = "Positive" if gru == 1 else "Negative"
    same = "Yes" if lstm_label == gru_label == expected_label else "No"
    print(f"{review:<35} {expected_label:<10} {lstm_label:<13} {gru_label:<12} {same}")

Output :

<img width="731" height="145" alt="image" src="https://github.com/user-attachments/assets/33f65668-8e1a-40fa-be37-473e2db2a364" />
