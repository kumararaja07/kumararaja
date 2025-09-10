import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

texts = [
    "I love this movie",
    "This film was terrible",
    "Amazing storyline and acting",
    "Worst movie ever",
    "I enjoyed the plot",
    "Not worth watching",
    "Fantastic direction",
    "Poorly made and boring",
    "Great performance",
    "I hated it"
]

labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=100)

X_train, X_test = X[:8], X[8:]
y_train, y_test = labels[:8], labels[8:]

model = Sequential([
    Embedding(input_dim=10000, output_dim=32, input_length=100),
    GRU(100, dropout=0.2, recurrent_dropout=0.2),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=2,
    validation_split=0.25,
    callbacks=[early_stop]
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
