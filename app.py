from flask import Flask, request, render_template #For API Implementation
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.svm import SVC
from transformers import pipeline # For text summarization
import joblib

app = Flask(__name__)


# Run model classification
model = joblib.load('svm_model.pkl')
#Load Google News Word2Vec Model for Feature Extraction
w2v_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/validate', methods=['POST'])
def validate():
    text = request.form['text']

    # Policies input classification
    problem_list = []
    paragraphs = split_into_paragraphs(text)
    # print("Paragraphs: ",paragraphs)

    for paragraph in paragraphs:
        prediction = classify_policy(paragraph)
        print("Prediction: ",prediction, " - Paragraph: ",paragraph)
        if prediction == 0:
            problem_list.append(paragraph)

    print("Problematic list: ",problem_list)
    
    #Highlight violated policies 
    if len(problem_list) >= 1:
        text = highlight_problems(text, problem_list)

    # Text summarization 
    summary_list = get_summarization(problem_list)
    print(summary_list)
   
    # Render the result.html template with the validation results
    return render_template('result.html', text=text, problem_list=problem_list, summary_list=summary_list)


#Data Processing
def preprocessing_policy(policy):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
   
    policy = policy.lower()
    policy = re.sub('[%s]' % re.escape(string.punctuation), '', policy)
    policy = re.sub('\w*\d\w*', '', policy)
    tokens = word_tokenize(policy)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens


#Feature Extraction - Google News Word2Vec Model
def feature_extraction(df):
    embeddings = get_word2vec_embeddings(w2v_model, df)
    print("Done words embedded")
    return embeddings


#Create Vector Representations for Policies
def get_average_word2vec(preprocessed_datapoint, w2v_model, generate_missing=False, k=300):
    if len(preprocessed_datapoint)<1:
        return np.zeros(k)
    
    # Assign vector value if token is not in model: depends on generate_missing, = 0 in this case
    if generate_missing:
        vectorized = [w2v_model[token] if token in w2v_model else np.random.rand(k) for token in preprocessed_datapoint]
    else:
        vectorized = [w2v_model[token] if token in w2v_model else np.zeros(k) for token in preprocessed_datapoint]
    # Calculate the average vector of the datapoint 
    # by dividing sum of values in same axis to the number of token in a datapoint
    length_datapoint = len(vectorized)
    summed_vector = np.sum(vectorized, axis=0)
    averaged_vector = np.divide(summed_vector, length_datapoint)
    return averaged_vector

def get_word2vec_embeddings(vectors, data, generate_missing=False):
    embeddings = data['tokens'].apply(lambda x: get_average_word2vec(x, vectors, generate_missing=generate_missing))
    return list(embeddings)


#Text input classification
def classify_policy(policy_text):
    # Preprocess the policy
    tokens = preprocessing_policy(policy_text)
    df_tokens = pd.DataFrame({'tokens': [tokens]})
    embedding = feature_extraction(df_tokens)
    # Predict using SVM
    embedding_array = np.array(embedding)
    svm_prediction = model.predict(embedding_array.reshape(1, -1))[0]
    print("Done Model Classification")
    return svm_prediction

#Split policies to classify each policy
def split_into_paragraphs(document_content):
    # Normalize the line breaks
    normalized_content = document_content.replace('\r\n', '\n')
    # Split the document by double line breaks, which typically separate paragraphs
    policies = [p.strip() for p in normalized_content.split('\n\n') if p.strip()]
    return policies


#Highlight problems
def highlight_problems(text, problem_list):
    for problem in problem_list:
        # Replace newlines with spaces and then escape the entire string
        escaped_problem = re.escape(problem.replace('\n', ' '))
        # Create a regex pattern that accounts for variable whitespace
        pattern = escaped_problem.replace(r'\ ', r'\s+')
        highlighted = f'<span style="background-color: #ff0000;">{problem}</span>'
        text = re.sub(pattern, highlighted, text)
    return text.replace('\n', '<br>')


#text summarization
def get_summarization(problem_list):
    # Implement text summarization logic using bart
    bart = pipeline("summarization", model="facebook/bart-large-cnn", min_length=17, max_length=42)
    summary_list=[]
    
    for problem in problem_list:
        result = bart(problem)  
        summary_list.append(result)
  
    summary_list = [item['summary_text'] for sublist in summary_list for item in sublist]
    print(summary_list)
    return summary_list

if __name__ == '__main__':
    app.run(debug=True)
