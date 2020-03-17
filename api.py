
import os
import config
import flask
import stomp
import gc

import numpy as np
import pandas as pd
import traceback
import pickle
import logging

from threading import Thread

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel 
from sklearn.neighbors import NearestNeighbors

from scipy.sparse import csr_matrix

from flask import current_app, Blueprint, render_template, jsonify, redirect, render_template, request, current_app
from flask_cors import cross_origin

from extensions import db, make_celery
from models import Element, Activity
from funcs.utils import make_activity_from_message, make_element_from_message, handle_message

api = Blueprint('api', __name__, url_prefix='/api')

# Create a common dictionary of received updates
global messages
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

@api.route('/0.1/query_db', methods=['POST'])
@cross_origin()
def query_db():
    error_response = jsonify({'error': 'could not process request'})

    try:
        form = request.form
        target_model_name = form.get('target_model_name', 'Element')
        target_model_name = 'Element'
        with current_app.app_context():
            step = 0
            if target_model_name == 'Element':
                target_model = Element
            elif target_model_name == 'Activity':
                target_model = Activity
            transactions = db.session.query(target_model).all()
            query_results = [x.convert_to_dict() for x in transactions]
        elements = query_results
        response = jsonify({'response': elements})
        return response
    except:
        traceback.print_exc()
        return error_response
    return error_response

@api.route('/0.1/train/content_filter', methods=['POST'])
@cross_origin()
def train_content_filter():
    error_response = jsonify({'error': 'could not process request'})
    try:
        form = request.form
        target_model_name = form.get('target_model_name', 'Element')
        target_model_name = 'Element'
        with current_app.app_context():
            step = 0
            if target_model_name == 'Element':
                target_model = Element
            elif target_model_name == 'Activity':
                target_model = Activity
            transactions = db.session.query(target_model).all()
            query_results = [x.convert_to_dict() for x in transactions]
        elements = query_results
        print('Calculating similarities ... ')
        df= pd.DataFrame.from_records(elements, columns=['element_id', 'title', 'tags'])
        def transform_tag_string(tags):

            # Transforms tags to enable feature extraction
            if isinstance(tags, str):
                tags = tags.lower()
                tags = ' '.join(tags.split(','))
                tags = tags.replace('  ', ' ')
                tags = ''.join([x for x in tags if not x.isdigit()])
            return tags

        df['tags'] = df['tags'].apply(lambda x: transform_tag_string(x))
        df = df.dropna()
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words=['и', 'или'])
        tfidf_matrix = tf.fit_transform(df['tags'].values.astype(str))
        cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
        with open(model_path + "cosine_similarities.pkl", 'wb') as file:
            pickle.dump(cosine_similarities, file)
            print('Similarities saved')
        
        print('Filtering results ...')

        filtering_results = {}

        def process_row(idx, row):
            print(row['element_id'])
            similar_indices = cosine_similarities[idx].argsort()[:-100:-1] 
            similar_items = [(cosine_similarities[idx][i], df['element_id'].iloc[i]) for i in similar_indices] 
            filtering_results[row['element_id']] = similar_items[1:]

        threads = []
        for idx, row in df.iterrows():
            process = Thread(target=process_row, args=(idx, row))
            threads.append(process)
        
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()
        
        with open(model_path + "filtering_results.pkl", 'wb') as file:
            pickle.dump(filtering_results, file)
            print('Results saved')
        response = jsonify({'response': 'similarities found'})
        return response
    except:
        traceback.print_exc()
        return error_response
    return error_response

@api.route('/0.1/recommend/content_filter', methods=['POST'])
@cross_origin()
def recommend_content_filter():

    # Create an API endpoint for the content filtering system
    error_response = jsonify({'error': 'could not process request'})
    try:
        if request.method == 'POST':
            form = request.form
            element_id = int(form.get('element_id', 0))
            num_recs = int(form.get('num_recs', 1))
            with open(model_path + "filtering_results.pkl", 'rb') as file:
                filtering_results = pickle.load(file)
            recs = filtering_results[element_id][:num_recs]
            outputs = []
            for rec in recs: 
                outputs.append({'id': int(rec[1]), 'score': rec[0]})
            count = len(outputs)
            response = jsonify({'response': {'count': len(outputs), 'outputs': outputs}})
            try:
                return response
            except: 
                return error_response
    except:
        traceback.print_exc()
        return error_response


@api.route('/update/<message_type>', methods=['POST'])
@cross_origin()
def message_update(message_type):

    # Create an API endpoint for REST message data
    error_response = jsonify({'error': 'could not process request'})
    try:
        if not messages.get(message_type, None):
            messages[message_type] = []
        content = request.json
        package = handle_message(content, message_type)
        with current_app.app_context():
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
                            element_entry = Element(element_id=package['content']['element_id'], title="", tags="")
                            db.session.add(element_entry)
                    db.session.commit()
            messages[message_type].append(content)
        return jsonify({'response': 'message received'})
    except:
        traceback.print_exc()
        return error_response

@api.route('/update_checker', methods=['POST'])
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
