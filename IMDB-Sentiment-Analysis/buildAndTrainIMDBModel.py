import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

INSPECT_DATA = False

# Print dataset shapes and sample rows
def inspect_dataset(dataset, name):
    print(f"--- {name} Dataset ---")
    for text, label in dataset.skip(3).take(3):
        print("Text:", text.numpy().decode('utf-8')[:100], "...")
        print("Label:", "Positive" if label.numpy() == 1 else "Negative")
        print()

# Preprocess the dataset
def preprocess_dataset(dataset, tokenizer):
    texts = [text.numpy().decode('utf-8') for text, _ in dataset]
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=256, padding='post', truncating='post')
    labels = [label.numpy() for _, label in dataset]
    return padded_sequences, tf.convert_to_tensor(labels)

def main():
    # Load the IMDB dataset
    dataset, info = tfds.load('imdb_reviews', split=['train[:80%]', 'train[80%:90%]'], as_supervised=True, with_info=True)
    train_data, validation_data = dataset

    if INSPECT_DATA:
        inspect_dataset(train_data, "Training")
        inspect_dataset(validation_data, "Validation")

    # Initialize the tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
    tokenizer.fit_on_texts([text.numpy().decode('utf-8') for text, _ in train_data])

    train_texts, train_labels = preprocess_dataset(train_data, tokenizer)
    validation_texts, validation_labels = preprocess_dataset(validation_data, tokenizer)

    model = Sequential([
        Embedding(input_dim=10000, output_dim=16, input_length=256),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint_callback = ModelCheckpoint(filepath='model_checkpoint.h5', save_best_only=True, monitor='val_accuracy')

    model.fit(
        train_texts, 
        train_labels, 
        epochs=20, 
        validation_data=(validation_texts, validation_labels),
        callbacks=[checkpoint_callback]
    )

    model.save('savedModel')

if __name__ == '__main__':
    main()

