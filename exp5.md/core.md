import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

texts = [
    "I loved the movie, fantastic!",
    "Worst film ever, boring.",
    "It was okay, not great.",
    "Absolutely wonderful experience",
    "Terrible acting and bad plot",
    "Enjoyed every moment",
    "Not my cup of tea",
    "Brilliant and emotional",
    "Disappointing and dull",
    "Pretty average"
]

labels = np.array([1, 0, 0, 1, 0, 1, 0, 1, 0, 0])  

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=20)

X_train, X_test = X[:8], X[8:]
y_train, y_test = labels[:8], labels[8:]

model = Sequential([
    Embedding(input_dim=10000, output_dim=32, input_length=20),
    GRU(64, dropout=0.2, recurrent_dropout=0.2),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
model.fit(X_train, y_train, epochs=10, batch_size=2, validation_split=0.25, callbacks=[early_stop])

pred_probs = model.predict(X_test)
pred_labels = (pred_probs > 0.5).astype(int).flatten()

print(f"{'Review Text':<35} {'Actual':<10} {'Predicted':<10} {'Correct'}")
print("-" * 65)
for review, actual, predicted in zip(texts[8:], y_test, pred_labels):
    correct = "Y" if actual == predicted else "N"
    actual_label = "Positive" if actual == 1 else "Negative"
    predicted_label = "Positive" if predicted == 1 else "Negative"
    print(f"{review:<35} {actual_label:<10} {predicted_label:<10} {correct}")

Output :

<img width="930" height="276" alt="image" src="https://github.com/user-attachments/assets/b5571b66-d44e-40c1-88f1-2957977a147a" />
