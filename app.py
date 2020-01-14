import os
import config
import flask
import stomp

import numpy as np
import pandas as pd
import traceback
import pickle

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
# If ActiveMQ server works only on the host machine, Docker container must be launched with '--net=host' parameter to access port 61613
element_conn = stomp.Connection(host_and_ports=host_and_ports)
element_listener = CSVDataListener()
element_conn.set_listener('', element_listener)
element_conn.start()
element_conn.connect('admin', 'password', wait=True)

# Subscribe STOMP listener to a given destination
element_conn.subscribe(destination='/queue/recomendation_update', id=1, ack='client')

# Create a STOMP listener for activities using code above as a template
activities_conn = stomp.Connection(host_and_ports=host_and_ports)
activities_listener = CSVDataListener()
activities_conn.set_listener('', activities_listener)
activities_conn.start()
activities_conn.connect('admin', 'password', wait=True)

activities_conn.subscribe(destination='/queue/recomendation_activities', id=1, ack='client')

def make_activity_from_message(message):
    activity = {'id': message.get('element', None), 'user_id': message.get('userId', None), 'rating': message.get('weight'), 'status': None}
    return activity

def make_element_from_message(message):
    element = {'id': message.get('element', None), 'title': message.get('name', None), 'tags': message.get('tagsString', None)}
    try:
        element['tags'] = element['tags'].strip('"')
    except:
        pass
    return element

def init_dataset(limit=0):

    # This function reads CSV data and loads datasets into memory. It returns two preprocessed dataframes (from book_names.csv and bookmarks1m.csv) and a matrix representation of user book ratings
    df = pd.read_csv('./dataset/book_names.csv', sep=';', na_filter=True, error_bad_lines=False, names=['id', 'title', 'tags'], skiprows=1)
    
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
    
    return df, df_marks

def extract_books_x_users(df_marks):
    df_marks_clean = (df_marks[df_marks['rating'] != 0])
    df_marks_clean = df_marks_clean.drop(['status'],1).drop_duplicates()
    df_marks_users = df_marks_clean.sort_values(by=['user_id'])
    df_book_features = df_marks_users.pivot(index='book_id', columns='user_id', values='rating').fillna(0)
    mat_book_features = csr_matrix(df_book_features.values)
    return df_book_features, mat_book_features

df, df_marks  = init_dataset(limit=0)
df_book_features, mat_book_features = extract_books_x_users(df_marks)

print('Dataset initialized')

if not os.path.isfile('cosine_similarities.pkl'):
    # Find similarities between books using their tags
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words=['и', 'или'])
    tfidf_matrix = tf.fit_transform(df['tags'].values.astype(str))

    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix) 
    
    with open("cosine_similarities.pkl", 'wb') as file:
        pickle.dump(cosine_similarities, file)
else:
    with open("cosine_similarities.pkl", 'rb') as file:
        cosine_similarities = pickle.load(file)


results = {}
for idx, row in df.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1] 
    similar_items = [(cosine_similarities[idx][i], df['id'][i]) for i in similar_indices] 
    results[row['id']] = similar_items[1:]

print('Similarities found')

if not os.path.isfile('model_knn.pkl'):
    # Initialize kNN with problem-appropriate parameters
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    model_knn.fit(mat_book_features)

    with open("model_knn.pkl", 'wb') as file:
        pickle.dump(model_knn, file)
else:
    with open("model_knn.pkl", 'rb') as file:
        model_knn = pickle.load(file)

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

@app.route('/api/0.1/content_filter', methods=['POST'])
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

@app.route('/api/0.1/knn_recommender', methods=['POST'])
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

@app.route('/api/wip/model_updater', methods=['POST'])
@cross_origin()
def model_updater():

    # Update models with new CSV data
    error_response = jsonify({'error': 'could not process request'})
    try:
        status = {}

        df, df_marks  = init_dataset(limit=0)
        df_book_features, mat_book_features = extract_books_x_users(df_marks)

        print('Dataset initialized')
        status['base_dataset'] = 'ok'

        if not os.path.isfile('cosine_similarities.pkl'):
            # Find similarities between books using their tags
            tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words=['и', 'или'])
            tfidf_matrix = tf.fit_transform(df['tags'].values.astype(str))

            cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix) 
            
            with open("cosine_similarities.pkl", 'wb') as file:
                pickle.dump(cosine_similarities, file)
        else:
            with open("cosine_similarities.pkl", 'rb') as file:
                cosine_similarities = pickle.load(file)


        results = {}
        for idx, row in df.iterrows():
            similar_indices = cosine_similarities[idx].argsort()[:-100:-1] 
            similar_items = [(cosine_similarities[idx][i], df['id'][i]) for i in similar_indices] 
            results[row['id']] = similar_items[1:]

        status['filtering_system'] = 'ok'
        print('Similarities found')

        if not os.path.isfile('model_knn.pkl'):
            # Initialize kNN with problem-appropriate parameters
            model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
            model_knn.fit(mat_book_features)

            with open("model_knn.pkl", 'wb') as file:
                pickle.dump(model_knn, file)
        else:
            with open("model_knn.pkl", 'rb') as file:
                model_knn = pickle.load(file)

        status('knn_model') = 'ok'
        print('KNN model created')
        response = jsonify({'response': {'status': status}})
    except:
        return error_response

@app.route('/api/wip/message_checker', methods=['POST'])
@cross_origin()
def message_checker():

    # Create an API endpoint for testing STOMP messaging
    error_response = jsonify({'error': 'could not process request'})
    try:
        if request.method == 'POST':
            form = request.form
            clean_up_mode = form.get('clean_up', 'false')
            if clean_up_mode == 'true':
                clean_up_flag = True
            else:
                clean_up_flag = False
            out_messages = {}
            out_messages['element'] = {}
            out_messages['element']['count'] = len(element_listener.message_list)
            out_messages['element']['messages'] = element_listener.message_list
            out_messages['activities'] = {}
            out_messages['activities']['count'] = len(activities_listener.message_list)
            out_messages['activities']['messages'] = activities_listener.message_list
            # out_messages = base_listener.message_list
            if clean_up_flag:
                element_listener.message_list = []
                activities_listener.message_list = []
            count = len(out_messages)
            response = jsonify({'response': {'message_queues': out_messages}})
            try:
                return response
            except: 
                return error_response
    except:
        traceback.print_exc()
        return error_response

@app.route('/api/wip/csv_updater', methods=['POST'])
@cross_origin()
def csv_updater():

    # Create an API endpoint for testing STOMP messaging
    error_response = jsonify({'error': 'could not process request'})
    try:
        if request.method == 'POST':
            form = request.form
            clean_up_mode = form.get('clean_up', 'false')
            if clean_up_mode == 'true':
                clean_up_flag = True
            else:
                clean_up_flag = False
            if len(element_listener.message_list) > 0 or len(activities_listener.message_list) > 0:
                elements = []
                activities = []
                if len(element_listener.message_list) > 0:
                    element_messages = element_listener.message_list
                    elements = [make_element_from_message(x['message']) for x in element_messages]
                    # elements = list(np.unique(np.array(elements).astype(str)))
                if len(activities_listener.message_list) > 0:
                    activities_messages = activities_listener.message_list
                    activities = [make_activity_from_message(x['message']) for x in activities_messages]
                    # activities = list(np.unique(np.array(activities).astype(str)))

                # print('='*15)
                # print(elements)
                # print('-'*15)
                # print(activities)
                # print('='*15)

                global df
                global df_marks

                nu_df = pd.DataFrame(elements, columns=['id', 'title', 'tags']).drop_duplicates()
                nu_df_marks = pd.DataFrame(activities, columns=['book_id', 'user_id', 'rating', 'status']).drop_duplicates()
                df = df.append(nu_df)
                df_marks = df_marks.append(nu_df_marks)

                df.to_csv('./dataset/book_names.csv', sep=';', index=False)
                df_marks.to_csv('./dataset/bookmarks1m.csv',sep=';', index=False)
                # out_messages = base_listener.message_list
                if clean_up_flag:
                    element_listener.message_list = []
                    activities_listener.message_list = []
                
                response = jsonify({'response': 'CSV update finished'})
                try:
                    return response
                except: 
                    return error_response
            else:
                response = jsonify({'response': 'No messages in queues'})
    except:
        traceback.print_exc()
        return error_response



if __name__ == "__main__":

    # Use Flask development server to run the application with multithreading enabled
    app.run(host='0.0.0.0', port='5002', debug=True, threaded=True)