import os
import config
import flask
import stomp

import pandas as pd
import traceback

from flask import (Flask, session, g, json, Blueprint,flash, jsonify, redirect, render_template, request,
                   url_for, send_from_directory, send_file)
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel 
from sklearn.neighbors import NearestNeighbors

from scipy.sparse import csr_matrix

from stomp_receiver import CSVDataListener
from dotenv import load_dotenv

# Load environment variables from the .env file with python-dotenv module
load_dotenv()

# Initialize main Flask application and allow CORS
app = Flask(__name__)
cors = CORS(app)

# Load app config from config.py file, use config variable to point at STOMP/ActiveMQ host and ports
app.config.from_object(os.environ['APP_SETTINGS'])
host_and_ports = app.config['HOSTS_AND_PORTS']

# Create a STOMP listener bound to specified host and ports using imported class from stomp_receiver.py
conn = stomp.Connection(host_and_ports=host_and_ports)
base_listener = CSVDataListener()
conn.set_listener('', base_listener)
conn.start()
conn.connect('admin', 'password', wait=True)

# Subscribe STOMP listener to a given destination
conn.subscribe(destination='/queue/messages', id=1, ack='auto')

def init_dataset(path, limit=0):
    # This function reads CSV data and loads datasets into memory. It returns two preprocessed dataframes (from book_names.csv and bookmarks1m.csv) and a matrix representation of user book ratings
    df = pd.read_csv(path, sep=';', na_filter=True, error_bad_lines=False, names=['id', 'title', 'tags'], skiprows=1)
    
    def transform_tag_string(tags):

        # Transforms tags to enable feature extraction
        if isinstance(tags, str):
            tags = tags.lower()
            tags = ' '.join(tags.split(','))
            tags = tags.replace('  ', ' ')
            tags = ''.join([x for x in tags if not x.isdigit()])
        return tags

    df['tags'] = df['tags'].apply(lambda x: transform_tag_string(x))
    df = df.dropna(subset = ['id', 'title', 'tags'])

    # Only top N entries from book_names.csv are sent into the output dataframe if limit is specified
    if limit > 0:
        df = df[:limit]

    df_marks = pd.read_csv('./dataset/bookmarks1m.csv',sep=';', na_filter=True, error_bad_lines=False, names=['book_id', 'user_id', 'rating', 'status'], skiprows=1)
    df_marks_clean = (df_marks[df_marks['rating'] != 0])
    df_marks_clean = df_marks_clean.drop(['status'],1).drop_duplicates()
    df_marks_users = df_marks_clean.sort_values(by=['user_id'])
    df_book_features = df_marks_users.pivot(index='book_id', columns='user_id', values='rating').fillna(0)
    mat_book_features = csr_matrix(df_book_features.values)

    return df, df_book_features, mat_book_features

df, df_book_features, mat_book_features = init_dataset('./dataset/book_names.csv', limit=0)

print('Dataset initialized')

# Find similarities between books using their tags

tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words=['и', 'или'])
tfidf_matrix = tf.fit_transform(df['tags'].values.astype(str))

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix) 
results = {}
for idx, row in df.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1] 
    similar_items = [(cosine_similarities[idx][i], df['id'][i]) for i in similar_indices] 
    results[row['id']] = similar_items[1:]

print('Similarities found')

# Initialize kNN with problem-appropriate parameters

model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model_knn.fit(mat_book_features)

print('KNN model created')

def translate_indices_mat_to_df(indices):
    # This function translates index values from matrix representation to the actual values from dataframe

    indices=indices[0]
    translated_indices = []
    for idx in indices:
        translated_indices.append(df_book_features.index[idx])
    return translated_indices

def translate_idx_df_to_mat(idx):
    # This function performes the inverse translation of index values from dataframe to matrix

    df_indices = df_book_features.index.tolist()
    return df_indices.index(idx)

def item(book_id):  
    # Return book title by index

    return df.loc[df['id'] == book_id]['title'].tolist()[0]

def extract_filtered_recs(book_id, num):
    # Return a list of recommended similar books, each one represented as a dictionary with id, title and score

    recs = results[book_id][:num]
    outputs = []
    for rec in recs: 
        outputs.append({'id': int(rec[1]), 'title': item(rec[1]), 'score': rec[0]})
    return outputs

def extract_knn_recs(book_id, num):
    # Use initialized kNN to get a list of recommended books based on user rating patterns. Each item is represented by its index and distance from the target book

    outputs = []
    distances, indices = model_knn.kneighbors(
            mat_book_features[translate_idx_df_to_mat(book_id)],
            n_neighbors=10)
    distances = distances[0]
    indices = translate_indices_mat_to_df(indices)
    recs = zip(distances, indices)
    counter = 0
    for distance, idx in recs:
        if counter < num and idx != book_id:
            print(distance, idx)
            outputs.append({'id': int(idx), 'distance': distance})
            counter += 1
    return outputs

@app.route('/api/0.1/content_filter', methods=['POST', 'GET'])
@cross_origin()
def content_filter():
    # Create an API endpoint for the content filtering system

    error_response = jsonify({'error': 'could not process request'})
    try:
        if request.method == 'POST':
            form = request.form
            book_id = int(form['book_id'])
            num_recs = int(form['num_recs'])
            filtered_recs = extract_filtered_recs(book_id, num_recs)
            count = len(filtered_recs)
            response = jsonify({'response': {'count': count, 'recs': filtered_recs}})
            try:
                return response
            except: 
                return error_response
    except:
        traceback.print_exc()
        return error_response

@app.route('/api/0.1/knn_recommender', methods=['POST', 'GET'])
@cross_origin()
def knn_recommender():
    # Create an API endpoint for the kNN-based recommendation system

    error_response = jsonify({'error': 'could not process request'})
    try:
        if request.method == 'POST':
            form = request.form
            book_id = int(form['book_id'])
            num_recs = int(form['num_recs'])
            filtered_recs = extract_knn_recs(book_id, num_recs)
            count = len(filtered_recs)
            response = jsonify({'response': {'count': count, 'recs': filtered_recs}})
            try:
                return response
            except: 
                return error_response
    except:
        traceback.print_exc()
        return error_response

if __name__ == "__main__":
    # Use Flask development server to run the application with multithreading enabled
    
    app.run(host='0.0.0.0', port='5002', debug=True, threaded=True)