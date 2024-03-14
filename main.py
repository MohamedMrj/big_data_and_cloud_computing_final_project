# Import necessary libraries and modules
import nltk  # Natural Language Toolkit
nltk.download('vader_lexicon')  # Download VADER lexicon for sentiment analysis
nltk.download('punkt')  # Download Punkt Tokenizer models
nltk.download('stopwords')  # Download stopwords
nltk.download('wordnet')  # Download WordNet
from get_video_comments import video_comments  # Import function to get video comments
import csv  # Import module to work with CSV files
from preprocess import preprocess_text  # Import function to preprocess text
import model  # Import module containing the LSTM model
from visualize import plot_sentiment_distribution, plot_training_history, plot_confusion_matrix
from nltk.sentiment import SentimentIntensityAnalyzer  # Import SentimentIntensityAnalyzer class

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

def main():
    # Get video comments using the video_comments function
    comments = video_comments()

    # Open a text file in write mode with utf-8 encoding
    # Write each comment on a new line
    with open("comments.txt", "w", encoding="utf-8") as file:
        for comment in comments:
            file.write(comment + "\n")

    # Open the text file in read mode with utf-8 encoding
    # Read all lines into a list
    with open('comments.txt', 'r', encoding="utf-8") as file:
        data = file.readlines()

    # Open a CSV file in write mode with utf-8 encoding
    with open('comments.csv', 'w', newline="", encoding="utf-8") as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile)
        # Iterate over each comment in the data
        for comment in data:
            # Preprocess the comment (remove unnecessary characters, stop words, etc.)
            processed_comment = preprocess_text(comment.strip())
            # Write the processed comment to the CSV file
            writer.writerow([processed_comment])

    # Call the model training function from the model module
    # This function returns the training history, true labels, and predicted labels
    history, y_true, y_pred = model.train_lstm_model()

    # Visualizing the sentiment distribution
    # For each comment, we label it using VADER sentiment analysis
    original_labels = [label_with_vader(comment) for comment in comments]
    # Plot the distribution of the original labels
    plot_sentiment_distribution(original_labels, title='Original Sentiment Distribution')

    # Plotting training history
    # This includes loss and accuracy over each epoch
    plot_training_history(history)

    # Plotting confusion matrix
    # This shows how well the model performed on each class
    plot_confusion_matrix(y_true, y_pred, classes=['Neutral', 'Positive', 'Negative'])

# If this script is run (instead of imported), call the main function
if __name__ == "__main__":
    main()