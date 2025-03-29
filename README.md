# LSTM - Sentiment Analysis

A twitter sentiment analysis project built with LSTM.
<img width="641" alt="Screenshot 2025-03-26 at 6 54 16 PM" src="https://github.com/user-attachments/assets/713f466f-4ced-4e48-b1f9-6bdc7ea24d00" />
This is a deep learning project that deals with building an LSTM neural network to predict sentiments of given tweets. 

## Running the Streamlit App

1. **Install Dependencies**:  
   Run the following command to install required packages.  
   ```
   pip install -r requirements.txt
   ```
2. **Run the App**:  
   Launch the Streamlit application by executing:  
   ```
   streamlit run app.py
   ``` 

3. **View the App**:  
   A browser window should open showing the interactive sentiment analysis interface.


4. **Get the sentiment**:  
   Write the tweet and it will give you the sentiment of the tweet, along with the lime score for each word. The computation time might be longer for the lime score, depending on length of tweet. 


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


## Requirements

Make sure you have Python 3.7+ installed. The project uses:
- numpy, pandas, matplotlib, seaborn
- nltk, scikit-learn
- TensorFlow / Keras
- streamlit

### Model Building and Architecture
This multi-class LSTM model uses four LSTM layers with descending unit sizes and includes BatchNormalization, Dropout, and L2 regularization for stability and generalization. Key choices:

- **Layer Configuration**:
  - Four LSTM layers with 64, 48, 32, and 16 units respectively. Decreasing units help distill features and reduce overfitting. `return_sequences=True` for the first three layers to feed subsequent LSTM layers. 
- **Initializers**:
  - `GlorotUniform` (Xavier) for kernel weights balances variance and prevents gradient issues. It helps maintian initialization, by ensuring that the variance of activations remains stable by keeping it proportional across layers. By setting the initial variance properly, it will help ensure that gradients flow well during backpropagation, allowing the network to learn effectively.

  - `Orthogonal` for recurrent weights helps LSTMs maintain stable gradients over time. These weights help preserve the flow of information across time steps by preventing the gradients from vanishing or exploding during backpropagation. This ensures that the model can effectively capture the context and relationships between words, even in long sentences, leading to more accurate sentiment predictions. Orthogonal weights work by preserving the length (or norm) of the hidden state vector as it is repeatedly transformed by the recurrent weight matrix during each time step. 

  - `Zeros` for bias initialization. Using zeros for bias initialization in an LSTM model for sentiment analysis is a common practice because it provides a neutral starting point for the biases, ensuring that the model does not favor any particular activation or behavior at the beginning of training.

The model was tested with multiple of the below hyper-parameters, these tended to perform the best in terms of performance and computation time, balance. Therefore this approach was taken: 

- **Regularization**:
  - L2 regularization (λ = 1e-4) on each LSTM layer and the Dense layer to discourage large weights and reduce overfitting. 
- **Batch Normalization**:
  - Applied after each LSTM layer to normalize activations, stabilize learning, and reduce internal covariate shift.
- **Dropout**:
  - Dropout rates ranging from 20% to 30% across LSTM layers to prevent reliance on specific neurons and improve generalization.
- **Output Layer**:
  - A Dense layer with softmax activation to produce probabilities across three sentiment classes.
- **Compilation**:
  - Uses the Adam optimizer with a learning rate of 1e-4.
  - `categorical_crossentropy` loss function for multi-class classification.
  - Tracks AUC (Area Under the ROC Curve) as a metric.

### Model Performance

The LSTM was trained with:
- Batch size: 64
- Epochs: up to 100 (with early stopping monitoring validation loss)

At the end of training:
- AUC: 0.8061, accuracy: 0.6310, loss: 0.8693
- Validation AUC: 0.7768, validation accuracy: 0.6031, validation loss: 0.9254

#### Test Set Results

- Accuracy: ~64%

| Metric    | Class 0 (Negative) | Class 1 (Neutral) | Class 2 (Positive) |
| --------- | ------------------ | ----------------- | ------------------ |
| Precision | 0.60               | 0.41              | 0.61               |
| Recall    | 0.69               | 0.26              | 0.64               |
| F1-Score  | 0.64               | 0.32              | 0.63               |

**Confusion Matrix:**
```
[ [1098, 175, 329],
  [ 349, 224, 287],
  [ 398, 147, 976] ]
```

**Per-class AUC:**
| Class             | AUC    |
| ----------------- | ------ |
| Class 0 (Negative) | 0.7521 |
| Class 1 (Neutral)  | 0.6657 |
| Class 2 (Positive) | 0.7608 |

MCC (Matthews Correlation Coefficient): 0.3329

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

### Business Application

The sentiment analysis model for Twitter tweets can provide significant value to businesses by enabling them to:

- **Monitor Brand Sentiment**: Analyze customer opinions about their brand, products, or services in real-time to identify trends and address issues proactively.
- **Competitor Analysis**: Track sentiment around competitors to gain insights into their strengths and weaknesses.
- **Customer Feedback Analysis**: Understand customer satisfaction and pain points by analyzing tweets related to their offerings.
- **Marketing Campaign Evaluation**: Measure the impact of marketing campaigns by assessing the sentiment of tweets before, during, and after the campaign.
- **Crisis Management**: Detect negative sentiment spikes to respond quickly to potential PR crises.
- **Product Development**: Gather insights from customer feedback to guide product improvements or new feature development.

### Further Development of the App and Its Business Application

This application can be extended to allow businesses to upload a CSV file containing multiple tweets for batch sentiment analysis. This enhancement would enable businesses to:
- **Scale Analysis**: Process large datasets of tweets efficiently, saving time and resources.
- **Data-Driven Decision Making**: Use aggregated sentiment insights to make informed decisions about marketing strategies, customer engagement, and product development.
- **Trend Analysis**: Identify patterns and trends over time by analyzing historical tweet data.
- **Custom Reporting**: Generate detailed sentiment reports for stakeholders to support strategic planning.

By incorporating CSV input functionality, the app can become a more versatile tool for businesses, empowering them to leverage data for competitive advantage.




