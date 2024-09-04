import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


def main():

    # Load the IMDB dataset
    dataset = tfds.load('imdb_reviews', split=['train[:80%]','test'], as_supervised=True)
    train_data, test_data = dataset

    # Initialize the tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
    tokenizer.fit_on_texts([text.numpy().decode('utf-8') for text, _ in train_data])

    # Preprocess the dataset
    def preprocess_dataset(dataset):
        texts = [text.numpy().decode('utf-8') for text, _ in dataset]
        sequences = tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=256, padding='post', truncating='post')
        labels = [label.numpy() for _, label in dataset]
        return padded_sequences, tf.convert_to_tensor(labels)

    test_texts, test_labels = preprocess_dataset(test_data)

    model = tf.keras.models.load_model('savedModel')

    test_loss, test_accuracy = model.evaluate(test_texts, test_labels)

    # Print the results
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy * 100 : .4f}")


if __name__ == '__main__':
    main()