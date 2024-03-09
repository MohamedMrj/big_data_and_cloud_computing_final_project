
# YouTube Comment Sentiment Analysis
![Figure_1](https://github.com/MohamedMrj/big_data_and_cloud_computing_final_project/assets/113178714/89704d87-ed33-43f1-83db-727a368bb3aa)

## Figure_1:
`Original Sentiment Distribution (Figure 1):`
This bar chart represents the distribution of sentiments in the dataset. It shows how many comments are neutral (0), positive (1), and negative (2). The majority of comments are neutral, followed by a significant number of positive comments, and fewer negative comments.

![Figure_2](https://github.com/MohamedMrj/big_data_and_cloud_computing_final_project/assets/113178714/7f6723e7-f819-4390-a450-4d92beeb1b2e)

## Figure_2:
`Training and Validation Loss (Figure 2):`
This graph shows the model's loss on the training set and the validation set over epochs. The training loss is decreasing, which indicates the model is learning from the training data. However, the validation loss starts to increase after a certain point, which could suggest overfitting—meaning the model is learning the training data too well, including noise or patterns that don't generalize to new data.

![Figure_3](https://github.com/MohamedMrj/big_data_and_cloud_computing_final_project/assets/113178714/11ad0380-3be9-4aa7-9dbd-dc5c0f81d007)

## Figure_3:
`Confusion Matrix (Figure 3):`
The confusion matrix shows the performance of the model in terms of the true labels versus the predicted labels. The diagonal entries (top-left to bottom-right) represent correct predictions, while off-diagonal entries represent incorrect predictions. In this matrix, it appears that the model performs well in predicting neutral and positive sentiments but is less accurate with negative sentiments, possibly due to fewer training examples for that class.

![Figure_4](https://github.com/MohamedMrj/big_data_and_cloud_computing_final_project/assets/113178714/2115d84d-3781-4754-a1b3-4692d67513a3)

## Figure_4:
`Training and Validation Accuracy (Figure 4):`
This plot shows the accuracy of the model during training (how well it predicts the training data) versus validation (how well it predicts a subset of the data not used in training). The accuracy on the training set is fluctuating, which might indicate that the model is struggling to learn consistent patterns. The validation accuracy is generally lower than the training accuracy, which is expected, but it's also fluctuating, indicating potential issues with model capacity, overfitting, or the representativeness of the validation set.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Overview
This project aims to perform sentiment analysis on YouTube video comments. It involves fetching comments for a specific video, preprocessing the text data, training a Long Short-Term Memory (LSTM) neural network model to classify comments into positive, negative, or neutral sentiments, and visualizing the results.

## Project Structure
- `main.py`: The main script that orchestrates the comment fetching, preprocessing, model training, and visualization.
- `model.py`: Contains the LSTM model's definition, training, and evaluation logic.
- `preprocess.py`: Implements text preprocessing functions such as tokenization, stopwords removal, stemming, and lemmatization.
- `visualize.py`: Provides functions to visualize sentiment distribution, model training history, and a confusion matrix of predictions.
- `Components/get_video_comments.py`: A module to fetch comments from a specified YouTube video using the YouTube Data API.
- `README.md`: This file, provides an overview and instructions for the project.

## Dependencies
This project requires Python 3.8 or later, with the following packages:
- TensorFlow 2.x
- NLTK
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Requests (for YouTube Data API calls)

You will also need a valid YouTube Data API key to fetch comments from YouTube videos.

## Setup Instructions
1. **Python Installation**: Ensure Python 3.8 or later is installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

2. **Project Clone**: Clone this repository to your local machine by running
```bash git clone https://github.com/MohamedMrj/big_data_and_cloud_computing_final_project/```
in your terminal or command prompt.

4. **Install Dependencies**: Navigate to the root directory of the project in your terminal. Install the required Python packages by executing:

   ```bash
   pip install -r requirements.txt
   ```

   This command reads the `requirements.txt` file and installs all the packages listed there, ensuring you have all the necessary dependencies.

5. **API Key Configuration**: To fetch comments from YouTube videos, you need a valid YouTube Data API key. Follow the instructions provided by Google [here](https://developers.google.com/youtube/v3/getting-started) to obtain one. Once you have your API key, create a file named `creds.py` inside the `Components` directory. Add your API key to this file as follows:

   ```python
   API_KEY = 'your_api_key_here'
   ```

   Replace `'your_api_key_here'` with your actual YouTube Data API key.

6. **Run the Project**: With the dependencies installed and the API key configured, you're ready to run the project. Ensure you're still in the root directory of the project and execute:

   ```bash
   python main.py
   ```

   This command runs the `main.py` script, which fetches comments, processes them, trains the LSTM model, and generates a visualization of the sentiment analysis.

## Running the Project
1. Navigate to the root directory of the project in your terminal.
2. Run the main script with the command `python main.py`.
3. The script will fetch comments from the specified video, preprocess them, train the LSTM model, and output the visualization results.

## Customization
You can analyze comments from different YouTube videos by modifying the video ID in the `get_video_comments` function call within `main.py`.

## Contributing
Contributions to this project are welcome. Please fork the repository, make your changes, and submit a pull request.

## License
This project is open-sourced under the MIT License. See the LICENSE file for more details.

---
