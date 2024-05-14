import boto3
import json
import sys
import pandas as pd
from awsglue.utils import getResolvedOptions
from io import StringIO
import joblib
import torch
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from io import BytesIO


from collections import Counter

MAX_LENGTH = 128
TRAINING_BATCH_SIZE = 32
VALIDATION_BATCH_SIZE = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import spacy
import tqdm
from typing import List, Tuple

# Spacy підтримує багато мов. Зараз нам потрібна англійська
spacy_nlp = spacy.blank("en")


def tokenize_spacy(text: str) -> List[str]:
  """Tokenize string with SpaCy """

  tokens = spacy_nlp.tokenizer(text)
  return [str(token) for token in tokens]


import Stemmer
from typing import List

def stem(tokens: List[str]) -> List[str]:
  """Lower-case and stem tokens. """

  stemmer = Stemmer.Stemmer("english")
  tokens = [tok.lower() for tok in tokens]
  return stemmer.stemWords(tokens)



STOP_WORDS = stem(["the", "and", "a", "of", "to", "is", "in", "that", "this", "was", "as", "with", "for", "you", "are", "it"])

def remove_stop_words(tokens: List[str]) -> List[str]:
  return [token for token in tokens if token not in STOP_WORDS]

def preprocess(text: str) -> List[str]:
  tokens = tokenize_spacy(text)
  tokens = stem(tokens)
  tokens = remove_stop_words(tokens)
  return tokens
class Ensemble:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        predictions = []
        for model in self.models:
            if 'bert' in str(type(model)).lower():
                # Preprocess the input data for BERT
                model_name = "bert-base-uncased"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                X_encoded = tokenizer(X.tolist(),
                                      truncation=True,
                                      padding='max_length',
                                      max_length=MAX_LENGTH,
                                      return_tensors='pt')
                input_ids = X_encoded['input_ids'].to(device)
                attention_mask = X_encoded['attention_mask'].to(device)

                # Pass the input through the model
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask)
                    print(outputs.keys())
                logits = outputs.logits

                # Apply softmax to get probabilities
                probs = torch.nn.functional.softmax(logits, dim=1)

                # Convert probabilities to predicted class labels
                preds = torch.argmax(probs, dim=1)

                # Append predicted labels to the list
                predictions.append(preds.cpu().numpy())

            else:
                # For non-BERT models, make predictions directly
                X_test_processed = [preprocess(text) for text in X]
                tfidf_vectorizer = TfidfVectorizer()
                X_test_tfidf = tfidf_vectorizer.transform([' '.join(tokens) for tokens in X_test_processed])
                model_predictions = model.predict(X_test_tfidf)
                not_binary_count = sum(1 for p in model_predictions if p not in [0, 1])
                print("Number of predictions not 0 or 1:", not_binary_count)
                predictions.append(model_predictions)

        ensemble_predictions = []
        for preds in zip(*predictions):
            # For each example, count the votes for each class label
            votes = Counter(preds)
            # Take the class label with the most votes
            majority_vote = votes.most_common(1)[0][0]
            ensemble_predictions.append(majority_vote)

        return ensemble_predictions



# Get job parameters
args = getResolvedOptions(sys.argv, ['bucket_name', 'file_key'])

# Access parameters
bucket_name = args['bucket_name']
file_key = args['file_key']

# Create a boto3 client
s3 = boto3.client('s3')


# Read the JSON file from S3
response = s3.get_object(Bucket=bucket_name, Key=file_key)
data = json.loads(response['Body'].read().decode('utf-8'))
json_data= data['comments']
comments = [entry['comment'] for entry in json_data]

# Create DataFrame from comments
df = pd.DataFrame(comments)

df = df[['odsCode', 'commentText', 'visit', 'department']]
def extract_values(row):
    return row['month'], row['year']

# Apply the function to the column containing the dictionary
df[['month', 'year']] = df['visit'].apply(lambda x: pd.Series(extract_values(x)))
df.drop(columns=['visit'], inplace=True)
# Now your DataFrame should have separate columns for 'month' and 'year'
month_map = {
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December'
}
df['month'] = df['month'].map(month_map)

logistic_regression_key = 'models/logistic_regression_model.pkl'
decision_tree_key = 'models/decision_tree_model.pkl'
bert_model_key = 'models/bert_model_torch.pth'

# Load logistic regression model from S3 bucket
response = s3.get_object(Bucket=bucket_name, Key=logistic_regression_key)
logistic_regression_model_bytes = response['Body'].read()
logistic_regression_model = joblib.load(BytesIO(logistic_regression_model_bytes))

# Load decision tree model from S3 bucket
response = s3.get_object(Bucket=bucket_name, Key=decision_tree_key)
decision_tree_model_bytes = response['Body'].read()
decision_tree_model = joblib.load(BytesIO(decision_tree_model_bytes))

# Load BERT model from S3 bucket
response = s3.get_object(Bucket=bucket_name, Key=bert_model_key)
bert_model_bytes = response['Body'].read()
bert_model = torch.jit.load(BytesIO(bert_model_bytes))

models = [bert_model, decision_tree_model, logistic_regression_model]
ensemble = Ensemble(models)

X = df['commentText']  # Features

# Make predictions on the test set
pred = ensemble.predict(X)

predicted_sentiment_labels = ['positive' if p == 1 else 'negative' for p in pred]

# Create a new DataFrame to store the predictions along with the original 'Org Name' data
predictions_df = pd.DataFrame({'comment': X, 'Org Name': df.loc[X.index, 'odsCode'], 'Department': df.loc[X.index, 'department'],
                                'month': df.loc[X.index, 'month'],	'year': df.loc[X.index, 'year'],
                               'predicted_sentiment': predicted_sentiment_labels})

csv_buffer = StringIO()
predictions_df.to_csv(csv_buffer, index=False)
file_key = '../files/result.csv'
s3_path = f's3://{bucket_name}/{file_key}'

# Write CSV data to S3
s3_resource = boto3.resource('s3')
s3_resource.Object(bucket_name, file_key).put(Body=csv_buffer.getvalue())

print(f'DataFrame saved to: {s3_path}')
