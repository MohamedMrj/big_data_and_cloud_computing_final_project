# Import necessary libraries and modules
import numpy as np  # For numerical operations
import csv  # For reading/writing CSV files
from sklearn.model_selection import train_test_split  # For splitting the dataset
from sklearn.metrics import classification_report  # For evaluating the model
from sklearn.utils import class_weight  # For handling class imbalance
from keras.models import Sequential  # For creating a sequential model
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D  # For creating layers in the model
from keras.preprocessing.text import Tokenizer  # For tokenizing text
from keras.preprocessing.sequence import pad_sequences  # For padding sequences
from keras.utils import to_categorical  # For converting labels to categorical
from nltk.sentiment import SentimentIntensityAnalyzer  # For sentiment analysis
from preprocess import preprocess_text  # For preprocessing text

def label_with_vader(comment):
    """Label comments based on VADER sentiment analysis."""
    sid = SentimentIntensityAnalyzer()  # Instantiate SentimentIntensityAnalyzer
    scores = sid.polarity_scores(comment)  # Get polarity scores for the comment
    # Classify comment as positive, negative, or neutral based on compound score
    if scores['compound'] > 0.05:
        return 1  # Positive
    elif scores['compound'] < -0.05:
        return 2  # Negative
    else:
        return 0  # Neutral

def train_lstm_model():
    """Train an LSTM model for sentiment analysis."""
    # Open the CSV file containing the comments
    with open('comments.csv', 'r', newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)  # Create a CSV reader
        data = list(reader)  # Convert the reader to a list

    # Extract the comments from the data
    comments = [row[0] for row in data]
    # Preprocess the comments
    processed_comments = [preprocess_text(comment) for comment in comments]
    # Label the comments using VADER sentiment analysis
    labels = [label_with_vader(comment) for comment in processed_comments]

    # Create a tokenizer with a vocabulary of 5000 words
    tokenizer = Tokenizer(num_words=5000)
    # Fit the tokenizer on the processed comments
    tokenizer.fit_on_texts(processed_comments)
    # Convert the comments to sequences of integers
    X = tokenizer.texts_to_sequences(processed_comments)
    # Pad the sequences so they all have the same length
    X = pad_sequences(X, padding='post')

    # Convert labels list to a numpy array
    y = np.array(labels)
    # Convert labels to one-hot encoding
    y = to_categorical(y, num_classes=3)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a sequential model
    model = Sequential()
    # Add an Embedding layer with input dimension of 5000 and output dimension of 120
    model.add(Embedding(input_dim=5000, output_dim=120, input_length=X.shape[1]))
    # Add a SpatialDropout1D layer with dropout rate of 0.4
    model.add(SpatialDropout1D(0.4))
    # Add an LSTM layer with 256 units and dropout rate of 0.2
    model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
    # Add a Dense output layer with 3 units (for 3 classes) and softmax activation
    model.add(Dense(3, activation='softmax'))

    # Compile the model with categorical crossentropy loss, Adam optimizer, and accuracy metric
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Calculate class weights to handle class imbalance
    classes = np.unique(np.argmax(y_train, axis=1))
    class_weights_array = class_weight.compute_class_weight('balanced', classes=classes, y=np.argmax(y_train, axis=1))
    class_weight_dict = {classes[i]: class_weights_array[i] for i in range(len(classes))}

    # Train the model using the class weights
    history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test), class_weight=class_weight_dict)

    # Generate predictions on the test set
    y_pred = np.argmax(model.predict(X_test), axis=1)
    # Convert the one-hot encoded y_test to label encoded
    y_true = np.argmax(y_test, axis=1)

    # Return the training history, true labels, and predicted labels
    return history, y_true, y_pred
