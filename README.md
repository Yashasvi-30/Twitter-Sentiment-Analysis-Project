# 🐦 Twitter Sentiment Analysis using LSTM

This project is a **Twitter Sentiment Analysis** web application built using **Streamlit** and powered by an **LSTM (Long Short-Term Memory)** neural network. It classifies tweets into three categories: **Positive**, **Neutral**, and **Negative**.

The LSTM model is trained on a dataset from [Kaggle: Twitter Sentiment Dataset](https://www.kaggle.com/datasets/saurabhshahane/twitter-sentiment-dataset/data).

---

## 🧠 Model Overview

- **Model**: LSTM-based sentiment classifier built with Keras and TensorFlow
- **Data**: Twitter Sentiment Dataset (`Twitter_Data.csv`)
- **Preprocessing**: Includes cleaning tweets (removal of links, usernames, hashtags, punctuation, stopwords, etc.)
- **Labels**:
  - `-1`: Negative  
  - `0`: Neutral  
  - `1`: Positive

---

## 🖥️ Web App

The application is built with **Streamlit** and allows users to:

- Enter any tweet or short sentence.
- View the predicted sentiment using the pre-trained LSTM model.
- See results styled beautifully inside a clean container with a background image.

---

## 🗂️ Project Structure

TWITTER_SENTIMENT_ANALYSIS/ │ ├── data/ │ └── Twitter_Data.csv # Dataset file │ ├── utils/ │ ├── text_cleaner.py # Custom text cleaning functions │ ├── download_nltk.py # Auto-downloads stopwords for NLTK │ └── pycache/ # Compiled Python files │ ├── app.py # Streamlit app main file ├── train_model.py # LSTM training script ├── predict.py # Model prediction functions ├── tokenizer.pkl # Saved tokenizer used for inference ├── lstm_sentiment_model.keras # Trained LSTM model file ├── requirements.txt # Python dependencies └── README.md # Project documentation

---

## ⚙️ Installation & Setup

### 🔧 Prerequisites

- Python 3.7+
- pip (Python package installer)

### 💻 Clone the repository

To get started with the project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/Twitter-Sentiment-Analysis-Project.git
   cd Twitter-Sentiment-Analysis-Project

2. Install the required dependencies:
   ```bash
       pip install -r requirements.txt
3. Download the NLTK resources (stopwords):
   ```bash
      python download_nltk.py
   
### 🚀 Running the App

To run the Streamlit app:
       -1. Download Dependencies 
       -2. Train the Model
       -3. Start the Streamlit app:
           ```bash
              streamlit run app.py

          
The app will open in your default browser at http://localhost:8501/.


## 🧪 Example Predictions

Here are a few example tweets and their predicted sentiments:

| Tweet                                              | Predicted Sentiment |
|----------------------------------------------------|---------------------|
| "I love the new design of your website!"           | Positive            |
| "Nothing special about the event."                 | Neutral             |
| "Worst service I've ever experienced."             | Negative            |

## 🖼️ UI Preview

The Streamlit app has a clean and minimal interface with:
- A light-themed background with text placed in a dark transparent container for readability.
- A footer with author info and a GitHub link.

## 📌 Features

- Real-time sentiment prediction using a trained LSTM model.
- Clean, minimal, and user-friendly interface.
- Custom background styling using HTML and CSS in Streamlit.

## 📄 Dataset

- **Source:** [Kaggle: Twitter Sentiment Dataset](https://www.kaggle.com/datasets/saurabhshahane/twitter-sentiment-dataset/data).
- **Format:** CSV
- **Columns:**
  - `text`: The tweet content.
  - `airline_sentiment`: Sentiment label (Negative, Neutral, Positive).

## 🧑‍💻 Model Training

The model is trained using the Long Short-Term Memory (LSTM) architecture. The training script `train_model.py` processes the dataset, tokenizes the text, and trains the model.  
To train the model, run:
   ```bash
      python train_model.py```

After training, the model is saved in the `lstm_sentiment_model.keras` file and the tokenizer is saved as `tokenizer.pkl` for inference.
---

## 📑 Requirements

The project requires the following Python libraries, listed in `requirements.txt`:
- streamlit
- nltk
- pandas
- tensorflow
- scikit-learn
- matplotlib

## 📚 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📬 Contact

- **Author:** Your Name
- **GitHub:** [https://github.com/your-username](https://github.com/your-username)
- **Email:** your.email@example.com
