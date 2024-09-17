# Fake News Detection using Natural Language Processing (NLP)

## Overview

This project focuses on building a system to automatically detect **fake news** using Natural Language Processing (NLP) techniques. The project aims to classify news articles or statements into two categories: **fake** or **real**, leveraging machine learning models trained on textual data. Fake news detection is a crucial problem, especially in today's world of rapid information dissemination, where misinformation can have far-reaching consequences.

The system goes through various stages such as data preprocessing, feature extraction, model building, and evaluation, to provide an accurate classification of news.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Technologies Used](#technologies-used)
6. [Data](#data)
7. [Models](#models)
8. [Results](#results)
9. [Contributing](#contributing)
10. [License](#license)

## Project Structure

```bash
fake-news-detection/
│
├── data/
│   ├── train.csv                # Training dataset
│   ├── test.csv                 # Testing dataset
│   └── processed_data.csv        # Cleaned and preprocessed data
│
├── models/
│   ├── fake_news_detector.pkl    # Saved model
│   ├── model_training.ipynb      # Notebook for model training and evaluation
│
├── scripts/
│   ├── preprocess.py             # Data preprocessing scripts
│   ├── train_model.py            # Model training script
│   ├── predict.py                # Script for making predictions
│
├── notebooks/
│   └── EDA.ipynb                 # Exploratory Data Analysis notebook
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── main.py                       # Entry point for running the system
```

## Features

- **Text Preprocessing**: Tokenization, stop-word removal, stemming, lemmatization, and vectorization (TF-IDF or Word2Vec).
- **Machine Learning Models**: Implements popular classification algorithms such as Logistic Regression, Random Forest, Support Vector Machines (SVM), and Deep Learning models like LSTM.
- **Real-Time Prediction**: Ability to make real-time predictions on new news articles.
- **Performance Metrics**: Evaluation metrics include accuracy, precision, recall, F1-score, and confusion matrix.
- **Customizable Pipeline**: The pipeline can be easily extended with additional preprocessing steps or alternative models.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/fake-news-detection.git
   cd fake-news-detection
   ```

2. **Install dependencies**:

   Make sure you have Python 3.7+ installed. Then, install the required Python packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download or prepare the dataset**:
   Place the dataset in the `data/` folder. Ensure that the data is in a CSV format with necessary columns such as "text" and "label".

## Usage

### 1. Preprocess the Data

To preprocess the data (cleaning, tokenizing, vectorizing, etc.), use the script:

```bash
python scripts/preprocess.py
```

This will output a cleaned dataset file in `data/processed_data.csv`.

### 2. Train the Model

To train the fake news detection model, run the following script:

```bash
python scripts/train_model.py
```

The trained model will be saved in the `models/` directory.

### 3. Make Predictions

You can make predictions on new news articles using the saved model:

```bash
python scripts/predict.py --input "Your news article text here"
```

This will output whether the news is real or fake based on the model's predictions.

### 4. Run the Main Program

Alternatively, you can run the main program to perform all the steps together:

```bash
python main.py
```

This script will preprocess the data, train the model, and evaluate the results.

## Technologies Used

- **Python**: Core programming language.
- **Natural Language Processing**: 
  - Libraries: `nltk`, `spaCy`, `scikit-learn`, `transformers`
  - Techniques: Tokenization, Lemmatization, Stop-word removal, TF-IDF, Word2Vec
- **Machine Learning**:
  - Models: Logistic Regression, Support Vector Machine, Random Forest, Naive Bayes, LSTM (Deep Learning).
  - Frameworks: `scikit-learn`, `TensorFlow`, `Keras`
- **Data Visualization**: `matplotlib`, `seaborn` for visualizing model performance.
- **Data Handling**: `pandas`, `numpy`
  
## Data

The dataset used in this project consists of labeled news articles. It typically contains two important columns:

- **Text**: The content of the news article.
- **Label**: Indicates whether the news is fake or real (binary classification: 0 for real, 1 for fake).

If using your own dataset, ensure it follows a similar format.

### Data Preprocessing Steps:
- **Text Cleaning**: Remove HTML tags, punctuation, numbers, etc.
- **Lowercasing**: Convert all text to lowercase.
- **Tokenization**: Splitting text into individual tokens (words).
- **Stop-Word Removal**: Remove common stop words like "the", "is", etc.
- **Stemming/Lemmatization**: Reduce words to their base form.
- **Vectorization**: Convert textual data into numerical vectors using techniques like TF-IDF or Word2Vec.

## Models

We experiment with the following models:

- **Logistic Regression**: A simple but effective classification model.
- **Random Forest**: An ensemble model that works well with imbalanced data.
- **Support Vector Machine (SVM)**: A model that performs well with high-dimensional data.
- **Long Short-Term Memory (LSTM)**: A neural network architecture particularly good for sequence-based data like text.

## Results

The model performance is evaluated using several metrics:

- **Accuracy**: Measures the overall correctness of the model.
- **Precision**: The ratio of correctly predicted fake news to all predicted fake news.
- **Recall**: The ratio of correctly predicted fake news to all actual fake news.
- **F1-Score**: Harmonic mean of precision and recall.
- **Confusion Matrix**: Displays the number of true positives, true negatives, false positives, and false negatives.

Results are visualized in the `notebooks/EDA.ipynb` file.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please fork the repository and create a pull request. You can also open an issue to discuss any feature requests or bugs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.




