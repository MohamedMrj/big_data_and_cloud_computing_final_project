# Import necessary libraries for visualization and metrics
import matplotlib.pyplot as plt  # For creating plots
import seaborn as sns  # For creating more attractive plots
from sklearn.metrics import confusion_matrix  # For creating a confusion matrix
import numpy as np  # For numerical operations

def plot_sentiment_distribution(labels, title='Sentiment Distribution'):
    """Plots the distribution of sentiments based on labels."""
    # Get unique sentiments and their counts
    sentiments, counts = np.unique(labels, return_counts=True)
    # Create a bar plot of sentiment counts
    sns.barplot(x=sentiments, y=counts)
    # Set the title of the plot
    plt.title(title)
    # Set the x and y labels of the plot
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    # Display the plot
    plt.show()

def plot_training_history(history):
    """Plots the training accuracy and loss over epochs."""
    # Extract accuracy, validation accuracy, loss, and validation loss from history
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    # Create a range of epochs
    epochs = range(1, len(acc) + 1)

    # Plot training accuracy over epochs
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    # Plot validation accuracy over epochs
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    # Set the title of the plot
    plt.title('Training and validation accuracy')
    # Add a legend to the plot
    plt.legend()

    # Create a new figure for the loss plot
    plt.figure()

    # Plot training loss over epochs
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # Plot validation loss over epochs
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # Set the title of the plot
    plt.title('Training and validation loss')
    # Add a legend to the plot
    plt.legend()

    # Display the plots
    plt.show()
def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion matrix', normalize=False):
    """
    Computes and plots the confusion matrix.
    
    Parameters:
    y_true (array-like): Ground truth (correct) target values.
    y_pred (array-like): Estimated targets as returned by a classifier.
    classes (list): List of labels to index the matrix.
    title (str): Title for the heatmap.
    normalize (bool): Whether to normalize the confusion matrix.

    Returns:
    None
    """
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # If normalize is True, normalize the confusion matrix
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create a heatmap of the confusion matrix
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', xticklabels=classes, yticklabels=classes)
    
    # Set the title of the heatmap
    plt.title(title)
    # Set the y label of the heatmap
    plt.ylabel('True label')
    # Set the x label of the heatmap
    plt.xlabel('Predicted label')
    
    # Display the plot
    plt.show()