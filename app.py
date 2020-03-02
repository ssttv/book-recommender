import os
import config
import flask
import stomp
import gc

import numpy as np
import pandas as pd
import traceback
import pickle

from flask import (Flask, session, g, json, Blueprint,flash, jsonify, redirect, render_template, request,
                   url_for, send_from_directory, send_file)
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel 
from sklearn.neighbors import NearestNeighbors

from scipy.sparse import csr_matrix

from stomp_receiver import CSVDataListener
from dotenv import load_dotenv

from handlers import handle_message

import memory_manager

# Load environment variables from the .env file with python-dotenv module
load_dotenv()

# Initialize main Flask application and allow CORS
app = Flask(__name__)
cors = CORS(app)

# Load app config from config.py file, use config variable to point at STOMP/ActiveMQ host and ports
app.config.from_object(os.environ['APP_SETTINGS'])
memory_percentage = app.config['MEMORY_PERCENTAGE']

db = SQLAlchemy(app)
migrate = Migrate(app, db)

import models
from models import Element, Activity

# Create a common dictionary of received updates
messages = {}

# Use data from volume if it exists
if os.path.exists('/vol/'):
    dataset_path = '/vol/dataset/'
    if not os.path.exists('/vol/models'):
        os.mkdir('/vol/models')
    model_path = '/vol/models/'
else: 
    dataset_path = './dataset/'
    model_path = './'

print('Selected dataset path: ' + dataset_path)
print('Selected model path: ' + model_path)

try:
    def make_activity_from_message(message):
        activity = {'book_id': message.get('element', None), 'user_id': message.get('userId', None), 'rating': message.get('weight'), 'status': None}
        return activity

    def make_element_from_message(message):
        element = {'id': message.get('element', None), 'title': message.get('name', None), 'tags': message.get('tagsString', None)}
        try:
            element['tags'] = element['tags'].strip('"')
        except:
            pass
        return element

    def records_from_db(target_model, page_size=100000):
        # records = []

        step = 0
        transactions = db.session.query(target_model).all()
        query_results = [x.convert_to_dict() for x in transactions]
        
        # Unfinished code for pagination and handling of large DB queries

        # for query_result in query_results:
        #     records.append(query_result.convert_to_dict())
        # while True:
        #     print("Page #{}".format(step))
        #     start, stop = page_size * step, page_size * (step+1)
        #     transactions = db.session.query(target_model).slice(start, stop).all()
        #     if transactions is None:
        #         break
        #     query_results = transactions
        #     for query_result in query_results:
        #         records.append(query_result.convert_to_dict())
        #     query_results = None
        #     print(len(transactions))
        #     if len(transactions) < page_size:
        #         break
        #     gc.collect()
        #     step += 1

        return query_results

    def init_datasets_from_records(limit=0):
        records_element = records_from_db(Element)
        records_activity = records_from_db(Activity)
        print('Number of elements: {}'.format(len(records_element)))
        print('Number of activities: {}'.format(len(records_activity)))

        df_elements = pd.DataFrame.from_records(records_element, columns=['element_id', 'title', 'tags'])
        def transform_tag_string(tags):

            # Transforms tags to enable feature extraction
            if isinstance(tags, str):
                tags = tags.lower()
                tags = ' '.join(tags.split(','))
                tags = tags.replace('  ', ' ')
                tags = ''.join([x for x in tags if not x.isdigit()])
            return tags

        df_elements['tags'] = df_elements['tags'].apply(lambda x: transform_tag_string(x))
        df_elements = df_elements.dropna(subset = ['element_id', 'title', 'tags'])
        df_activities = pd.DataFrame.from_records(records_activity, exclude=['activity_id'], columns=['activity_id', 'element_id', 'user_id', 'rating', 'status'])
        return df_elements, df_activities

    def extract_books_x_users(df_activities):
        df_activities_clean = (df_activities[df_activities['rating'] != 0])
        df_activities_clean = df_activities_clean.drop(['status'],1).drop_duplicates()
        df_activities_users = df_activities_clean.sort_values(by=['user_id'])
        df_book_features = df_activities_users.pivot(index='element_id', columns='user_id', values='rating').fillna(0)
        mat_book_features = csr_matrix(df_book_features.values)
        return df_book_features, mat_book_features

    df_elements, df_activities = init_datasets_from_records()
    print('All queries done')
    df_book_features, mat_book_features = extract_books_x_users(df_activities)
    print('Dataset initialized')

    # if not os.path.isfile(model_path + 'cosine_similarities.pkl'):
    #     # Find similarities between books using their tags
    #     tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words=['и', 'или'])
    #     tfidf_matrix = tf.fit_transform(df_elements['tags'].values.astype(str))

    #     cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix) 
        
    #     with open(model_path + "cosine_similarities.pkl", 'wb') as file:
    #         pickle.dump(cosine_similarities, file)
    # else:
    #     with open(model_path + "cosine_similarities.pkl", 'rb') as file:
    #         cosine_similarities = pickle.load(file)
    #     print("Similarities loaded")

    # Find similarities between books using their tags
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words=['и', 'или'])
    tfidf_matrix = tf.fit_transform(df_elements['tags'].values.astype(str))

    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix) 
    
    with open(model_path + "cosine_similarities.pkl", 'wb') as file:
        pickle.dump(cosine_similarities, file)

    results = {}
    for idx, row in df_elements.iterrows():
        similar_indices = cosine_similarities[idx].argsort()[:-100:-1] 
        similar_items = [(cosine_similarities[idx][i], df_elements['element_id'][i]) for i in similar_indices] 
        results[row['element_id']] = similar_items[1:]

    print('Similarities found')

    # if not os.path.isfile(model_path + 'model_knn.pkl'):
    #     # Initialize kNN with problem-appropriate parameters
    #     model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    #     model_knn.fit(mat_book_features)

    #     with open(model_path + "model_knn.pkl", 'wb') as file:
    #         pickle.dump(model_knn, file)
    # else:
    #     with open(model_path + "model_knn.pkl", 'rb') as file:
    #         model_knn = pickle.load(file)
    #     print("KNN model loaded")

    # Initialize kNN with problem-appropriate parameters
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    model_knn.fit(mat_book_features)

    with open(model_path + "model_knn.pkl", 'wb') as file:
        pickle.dump(model_knn, file)

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

    def item(element_id):  
        # Return book title by index
        return df_elements.loc[df_elements['element_id'] == element_id]['title'].tolist()[0]

    def extract_filtered_recs(element_id, num):

        # Return a list of recommended similar books, each one represented as a dictionary with id, title and score
        recs = results[element_id][:num]
        outputs = []
        for rec in recs: 
            outputs.append({'id': int(rec[1]), 'title': item(rec[1]), 'score': rec[0]})
        return outputs

    def extract_knn_recs(element_id, num):

        # Use initialized kNN to get a list of recommended books based on user rating patterns. Each item is represented by its index and distance from the target book
        outputs = []
        distances, indices = model_knn.kneighbors(
                mat_book_features[translate_idx_df_to_mat(element_id)],
                n_neighbors=10)
        distances = distances[0]
        indices = translate_indices_mat_to_df(indices)
        recs = zip(distances, indices)
        counter = 0
        for distance, idx in recs:
            if counter < num and idx != element_id:
                print(distance, idx)
                outputs.append({'id': int(idx), 'distance': distance})
                counter += 1
        return outputs
except:
    traceback.print_exc()
    print('Errors during initialization')

@app.route('/api/0.1/content_filter', methods=['POST'])
@cross_origin()
def content_filter():

    # Create an API endpoint for the content filtering system
    error_response = jsonify({'error': 'could not process request'})
    try:
        if request.method == 'POST':
            form = request.form
            element_id = int(form.get('element_id', 0))
            num_recs = int(form.get('num_recs', 1))
            filtered_recs = extract_filtered_recs(element_id, num_recs)
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
            element_id = int(form.get('element_id', 0))
            num_recs = int(form.get('num_recs', 1))
            filtered_recs = extract_knn_recs(element_id, num_recs)
            count = len(filtered_recs)
            response = jsonify({'response': {'count': count, 'recs': filtered_recs}})
            try:
                return response
            except: 
                return error_response
    except:
        traceback.print_exc()
        return error_response

@app.route('/api/update/<message_type>', methods=['POST'])
@cross_origin()
def message_update(message_type):

    # Create an API endpoint for REST message data
    error_response = jsonify({'error': 'could not process request'})
    try:
        if not messages.get(message_type, None):
            messages[message_type] = []
        content = request.json
        package = handle_message(content, message_type)
        if package.get('success', False):
            if message_type == 'element':
                element_entry = Element(element_id=package['content']['element_id'], title=package['content']['title'], tags=package['content']['tags'])
                db.session.merge(element_entry)
                db.session.commit()
            elif message_type == 'activity':
                activity_entry = Activity(element_id=package['content']['element_id'], user_id=package['content']['user_id'], rating=package['content']['rating'], status=package['content']['status'])
                if package['content']['element_id'] is not None:
                    known_element = db.session.query(Element).filter(Element.element_id==package['content']['element_id']).limit(10).all()
                    if known_element is not None and not len(known_element) > 0 :
                        element_entry = Element(element_id=package['content']['element_id'], title=None, tags=None)
                        db.session.add(element_entry)
                db.session.commit()
        messages[message_type].append(content)
        return jsonify({'response': 'message received'})
    except:
        traceback.print_exc()
        return error_response

@app.route('/api/update_checker', methods=['POST'])
@cross_origin()
def message_update_checker():
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
            
            global messages

            if messages.get('element', None):
                out_messages['element'] = {}
                out_messages['element']['count'] = len(messages.get('element'))
                out_messages['element']['messages'] = messages.get('element')
            if messages.get('activity', None):
                out_messages['activity'] = {}
                out_messages['activity']['count'] = len(messages.get('activity'))
                out_messages['activity']['messages'] = messages.get('activity')
            if messages.get('recalculate', None):
                out_messages['recalculate'] = {}
                out_messages['recalculate']['count'] = len(messages.get('recalculate'))
                out_messages['recalculate']['messages'] = messages.get('recalculate')
            if clean_up_flag:
                messages = []
            response = jsonify({'response': {'messages': out_messages}})
            try:
                return response
            except: 
                return error_response
    except:
        traceback.print_exc()
        return error_response

@memory_manager.managed_memory(percentage=memory_percentage)
def create_app(app):
    app.run(host='0.0.0.0', port='5002', debug=True, threaded=True)

if __name__ == "__main__":

    # Use Flask development server to run the application with multithreading enabled
    # app.run(host='0.0.0.0', port='5002', debug=True, threaded=True)
    create_app(app)