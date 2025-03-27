# LSTM - Sentiment Analysis

A twitter sentiment analysis project built with LSTM.

## Files Overview

- **README.md**: Project documentation.
- **prediction.ipynb**: Contains the full workflow:
  - Data import and exploratory data analysis (EDA)
  - Cleaning and pre-processing text data
  - Converting text to embeddings using GloVe and TF-IDF
  - Label encoding for sentiment classification
  - Building, training, and evaluating an LSTM model for multi-class sentiment prediction
- **app/**: Contains the Streamlit application to serve the model.

## Model Approach

The model builds a robust sentiment analysis pipeline:
- **Data Pre-processing**: Removing null values, duplicates, and unwanted columns.
- **Text Cleaning**: Using regex and NLTK for removing URLs, mentions, non-word characters, and lemmatization.
- **Embedding Tweets**: Combining GloVe embeddings with TF-IDF weighting to create numerical representations of tweets.
- **Model Building**: A multi-layer LSTM model with BatchNormalization, Dropout, and regularization to predict sentiment.
- **Evaluation**: Classification report, confusion matrix, ROC-AUC per class, and MCC score are used for assessment.

### Data Cleaning and Pre-processing Overview

- **Null Data Cleaning**: Removed any rows with null values and reset the dataframe index.
- **Pre-processing Text**: 
  - Removed URLs, non-word characters, hashtags (only the '#' symbol), non-ascii characters, and digits.
  - Fixed multiple spaces and trimmed leading/trailing spaces.
  - Applied lemmatization using NLTK's WordNetLemmatizer.
- **Embedding Text**: 
  - Loaded pre-trained 200-dimensional GloVe embeddings.
  - Applied TF-IDF vectorization to weight the importance of each word.
  - Combined the GloVe embeddings with TF-IDF scores to produce a robust numerical representation for each tweet.
- **Label Encoding**: Converted sentiment text labels into numerical classes.
- **Other Processing**: 
  - Adjusted data types and converted embedding vectors to a format suitable for feeding into the LSTM model.

## Running the Streamlit App

1. **Install Dependencies**:  
   Run the following command to install required packages.  
   ```
   pip install -r requirements.txt
   ```
2. **Run the App**:  
   Launch the Streamlit application by executing:  
   ```
   streamlit run app/app.py
   ``` 

3. **View the App**:  
   A browser window should open showing the interactive sentiment analysis interface.

## Requirements

Make sure you have Python 3.7+ installed. The project uses:
- numpy, pandas, matplotlib, seaborn
- nltk, scikit-learn
- TensorFlow / Keras
- streamlit

### Model Building and Architecture

This model uses a 70:20:10 split for train, validation, and test. Below is a richly detailed explanation of each design choice:

1. **Choice of LSTM**  
   LSTMs are effective for capturing long-range dependencies thanks to their gating mechanism, which controls how information flows and is retained or discarded. Tweets often contain sequential dependencies (e.g., negations or context words), so an LSTM architecture is well-suited for processing these nuances.

2. **Layer Configuration**  
   - The model has four LSTM layers with descending numbers of units (64 → 48 → 32 → 16). Decreasing the layer size helps the network progressively distill features while reducing overfitting risk.  
   - “return_sequences” is set on the first three LSTM layers to provide hidden states for subsequent LSTM layers, capturing richer temporal patterns before funneling to the next layer.

3. **Weight Initialization**  
   - **GlorotUniform (Xavier) Initializer**: Balances variance in weights across all layers, preventing gradients from becoming too large or too small. This supports faster convergence and more stable learning.  
   - **Orthogonal Recurrent Initializer**: Helps LSTM gates maintain stable gradients over many time steps, reducing vanishing or exploding gradient issues.

4. **Kernel Regularization (l2)**  
   - Each LSTM layer and the final Dense layer use L2 regularization (λ = 1e-4). This penalty term discourages strongly weighted connections, effectively reducing overfitting by smoothing the weight distribution.

5. **Batch Normalization**  
   - Placed after each LSTM layer to normalize activations before sending them to subsequent layers. By keeping the input distribution consistent, it allows the network to learn more reliably even at deeper levels.  
   - It can also have a mild regularizing effect, as it reduces the internal covariate shift during training.

6. **Dropout**  
   - Ranging from 20% to 30% across the LSTM layers. Randomly zeroing out neurons forces layers not to rely too heavily on specific nodes, thus improving generalization and mitigating overfitting.

7. **Dense Output Layer**  
   - Uses a softmax activation to produce probabilities across three sentiment classes.  
   - Also includes L2 regularization and GlorotUniform initialization to maintain the same stability and generalization benefits as the LSTM layers.

Overall, these carefully selected hyperparameters and initialization strategies reduce overfitting risks, stabilize training dynamics, and ensure effective learning from the sequential embeddings derived from tweets.

### Model Performance

The LSTM was trained with:
- Batch size: 64
- Epochs: up to 100 (with early stopping monitoring validation loss)

At the end of training:
- AUC: 0.7884, loss: 0.8895
- Validation AUC: 0.7809, validation loss: 0.9040

#### Test Set Results

- Accuracy: ~58%

Precision, recall, and F1-scores:
• Class 0 (Negative): precision 0.60, recall 0.70, F1 0.65  
• Class 1 (Neutral): precision 0.43, recall 0.24, F1 0.31  
• Class 2 (Positive): precision 0.61, recall 0.66, F1 0.63  

Confusion Matrix:
[ [1124, 139, 339],  
  [ 350, 205, 305],  
  [ 392, 132, 997] ]

Per-class AUC:
• Class 0: 0.7672  
• Class 1: 0.6860  
• Class 2: 0.7710  

MCC (Matthews Correlation Coefficient): 0.3422  

#### Interpreting the Scores
- **Precision**: Measures how often predicted positives were truly positive. For example, Class 2 (Positive) has a precision of 0.61, meaning that out of all tweets predicted as positive, 61% were actually positive.  
- **Recall**: Indicates the model’s ability to find all actual positives. Class 1 (Neutral) has a low recall of 0.24, suggesting many neutral tweets are overlooked.  
- **F1-Score**: The harmonic mean of precision and recall. For Class 0 (Negative), an F1 of 0.65 reflects a better balance between precision (0.60) and recall (0.70) compared to the neutral class.  
- **AUC**: Represents how well the model ranks positive classes above negative ones across thresholds. Per-class AUC scores (0.77 for Negative, 0.69 for Neutral, 0.77 for Positive) show moderate separability, with Neutral sentiment being the hardest to discriminate.  
- **MCC**: A robust measure accounting for true/false positives/negatives, with values between -1 and +1. The MCC of 0.34 suggests moderate correlation between predictions and true labels, indicating room for improved model alignment.

These metrics show that the model tends to handle positive and negative tweets more effectively than neutral ones, which often contain ambiguous or subtle sentiment cues that require more context. Overall, the results indicate reasonable performance, but highlight certain challenges such as class imbalance, ambiguous text, and data variability.

### Model Limitations

Although the model shows reasonable performance, several limitations remain:
- **Class Imbalances**: Lower metrics on the neutral class indicate uneven representation and difficulty in handling ambiguous sentiment.    
- **Data Constraints**: The variable nature of tweets and possible slang usage may require more robust language models or additional pre-processing to handle out-of-vocabulary terms effectively.
