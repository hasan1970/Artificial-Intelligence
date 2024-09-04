# IMDB Sentiment Analysis

This project focuses on building a machine learning model for binary sentiment classification using the IMDB movie reviews dataset. The model classifies reviews as either positive or negative, leveraging word embeddings and pooling techniques for efficient text processing.

## Technical Overview

### Key Aspects:
1. **Model Architecture**:
   - The model is a sequential neural network consisting of:
     - **Embedding Layer**: Converts the input sequences into dense vectors of fixed size.
     - **Global Average Pooling**: Reduces the dimensionality of the input, capturing key features.
     - **Dense Layers**: A fully connected layer with a sigmoid activation function for binary classification.

2. **Training**:
   - The model is trained using the **IMDB movie reviews dataset** from TensorFlow Datasets.
   - The dataset is tokenized and sequences are padded for uniform input length.
   - The model is trained with **binary crossentropy loss** and **Adam optimizer**, with validation performed on a subset of the dataset.
   - A **model checkpoint** saves the best-performing model based on validation accuracy.

3. **Reloadable Model**:
   - The trained model is saved and can be reloaded for evaluation or further use without retraining.

### Scripts

1. **buildAndTrainIMDBModel.py**: 
   - Loads and preprocesses the IMDB dataset.
   - Builds and trains a sentiment classification model using a simple machine learning architecture.
   - Implements checkpointing to save the model with the best validation performance.
   - The final model is saved for future evaluation or deployment.

2. **runModel.py**:
   - Reloads the saved model and evaluates its performance on the test dataset.
   - Reports accuracy and loss on the test data, demonstrating the model's effectiveness at sentiment classification.

## Results

- **Accuracy**: The model achieved an accuracy of **83.16%** on the test dataset. While the model provides a good baseline for sentiment classification, further tuning or architectural changes could improve its performance.

## Conclusion

This project demonstrates a machine learning approach to sentiment analysis using the IMDB movie reviews dataset. The model effectively processes text using embedding and pooling techniques, making it suitable for binary sentiment classification tasks. Its reusability through model saving and loading ensures flexibility for further experimentation or deployment.
