
# YouTube Comment Sentiment Analysis

## Overview
This project aims to perform sentiment analysis on YouTube video comments. It involves fetching comments for a specific video, preprocessing the text data, training a Long Short-Term Memory (LSTM) neural network model to classify comments into positive, negative, or neutral sentiments, and visualizing the results.

## Project Structure
- `main.py`: The main script that orchestrates the comment fetching, preprocessing, model training, and visualization.
- `model.py`: Contains the LSTM model's definition, training, and evaluation logic.
- `preprocess.py`: Implements text preprocessing functions such as tokenization, stopwords removal, stemming, and lemmatization.
- `visualize.py`: Provides functions to visualize sentiment distribution, model training history, and a confusion matrix of predictions.
- `Components/get_video_comments.py`: A module to fetch comments from a specified YouTube video using the YouTube Data API.
- `README.md`: This file, providing an overview and instructions for the project.

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

If you have a `requirements.txt` file in your project, it's important to mention it in the README file, especially in the setup instructions, to ensure users know how to install the necessary dependencies for your project. Since it seems you might already have setup instructions that mention installing dependencies via `requirements.txt`, I'll adjust the README template to highlight this file more explicitly.

Here's the revised section for the setup instructions in your README, considering the `requirements.txt`:

## Setup Instructions
1. **Python Installation**: Ensure Python 3.8 or later is installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

2. **Project Clone**: Clone this repository to your local machine by running `git clone <repository-url>` in your terminal or command prompt, where `<repository-url>` is the URL to this GitHub repository.

3. **Install Dependencies**: Navigate to the root directory of the project in your terminal. Install the required Python packages by executing:

   ```bash
   pip install -r requirements.txt
   ```

   This command reads the `requirements.txt` file and installs all the packages listed there, ensuring you have all the necessary dependencies.

4. **API Key Configuration**: To fetch comments from YouTube videos, you need a valid YouTube Data API key. Follow the instructions provided by Google [here](https://developers.google.com/youtube/v3/getting-started) to obtain one. Once you have your API key, create a file named `creds.py` inside the `Components` directory. Add your API key to this file as follows:

   ```python
   API_KEY = 'your_api_key_here'
   ```

   Replace `'your_api_key_here'` with your actual YouTube Data API key.

5. **Run the Project**: With the dependencies installed and the API key configured, you're ready to run the project. Ensure you're still in the root directory of the project and execute:

   ```bash
   python main.py
   ```

   This command runs the `main.py` script, which fetches comments, processes them, trains the LSTM model, and generates visualization of the sentiment analysis.

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

Remember to replace placeholder texts with specific details about your project, such as how to obtain the YouTube Data API key, specific commands to run for setting up the environment, and any additional steps needed to get the project running.