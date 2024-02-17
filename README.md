
# Text Summarizer

## Overview

This project provides a text summarization tool that utilizes the Transformer-based summarizer for condensing lengthy texts into concise summaries. Additionally, it incorporates the YAKE algorithm for keyword extraction and a Naive Bayes classifier for categorizing articles into topics such as business, sports, politics, etc.

## Features

- **Text Summarization:** Utilizes Transformer-based summarization techniques to generate summaries of input text.
- **Keyword Extraction:** Implements YAKE algorithm for extracting keywords from text.
- **Article Classification:** Employs a Naive Bayes classifier trained on various topics with an average accuracy of 94% to classify articles.
- **Recommendation System:** Recommends similar articles using the News API.
- **User Interface:** Users can interact with the application through a Streamlit-powered web interface by running `home.py`.

## How to Use

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/text-summarizer.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run home.py
   ```

4. Follow the instructions provided in the terminal to access the application in your web browser.

## Note

This project has not been deployed yet. Users can run the `home.py` file locally to access the functionality of the text summarizer.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request with your changes.

## Acknowledgments

- This project utilizes the following libraries:
  - [Hugging Face Transformers](https://github.com/huggingface/transformers) for text summarization.
  - [YAKE](https://github.com/LIAAD/yake) for keyword extraction.
  - [Streamlit](https://github.com/streamlit/streamlit) for the web interface.
  - [DataSet] (https://www.kaggle.com/datasets/rmisra/news-category-dataset/data) for training the classifier model.
    1.Misra, Rishabh. "News Category Dataset." arXiv preprint arXiv:2209.11429 (2022).
    2.Misra, Rishabh and Jigyasa Grover. "Sculpting Data for ML: The first act of Machine Learning." ISBN 9798585463570 (2021)..

---