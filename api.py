
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
from scipy import spatial

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, euclidean_distances

from sklearn.neighbors import NearestNeighbors

from scipy.sparse import csr_matrix

from flask import current_app, Blueprint, render_template, jsonify, redirect, render_template, request, current_app
from flask_cors import cross_origin

from extensions import Session, make_celery
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
            transactions = Session.query(target_model).all()
            query_results = [x.convert_to_dict() for x in transactions]
        elements = query_results
        response = jsonify({'response': elements})
        return response
    except:
        traceback.print_exc()
        return error_response
    return error_response


@api.route('/0.2/recommend/content_filter', methods=['POST'])
@cross_origin()
def recommend_content_filter_spark():

    # Create an API endpoint for the content filtering system
    error_response = jsonify({'error': 'could not process request'})
    try:
        if request.method == 'POST':
            form = request.form
            element_id = int(form.get('element_id', 0))
            num_recs = int(form.get('num_recs', 1))

            similarities = current_app.similarities
            out_similarities = similarities.where(similarities.element_source_id == element_id).orderBy('dot', ascending=False)
            out_ids = [row['element_target_id'] for row in out_similarities.collect()]

            
            # for out_id in out_ids[:5]:
            #     entry = (s_df_element.where(s_df_element.element_id == out_id))
            #     dot = out_similarities.where(out_similarities.element_target_id == out_id)
            #     if entry.head() is not None and dot is not None:
            #         print(entry.head().element_id, entry.head().title, dot.head().dot)
            # with open(model_path + "filtering_results.pkl", 'rb') as file:
            #     filtering_results = pickle.load(file)
            # recs = filtering_results[element_id][:num_recs]
            outputs = []
            for out_id in out_ids[:num_recs]:
                outputs.append({'id': int(out_id)})
            # for rec in recs: 
            #     outputs.append({'id': int(rec[1]), 'score': rec[0]})
            count = len(outputs)
            response = jsonify({'response': {'count': len(outputs), 'outputs': outputs}})
            try:
                return response
            except: 
                return error_response
    except:
        traceback.print_exc()
        return error_response


@api.route('/0.2/recommend/als_recommender', methods=['POST'])
@cross_origin()
def recommend_als_spark():

    # Create an API endpoint for the content filtering system
    error_response = jsonify({'error': 'could not process request'})
    try:
        if request.method == 'POST':
            form = request.form
            user_id = int(form.get('user_id', 0))
            num_recs = int(form.get('num_recs', 1))

            user_recs = current_app.user_recs
            target_df = user_recs[user_recs['user_id'] == user_id]
            target_recs = target_df['recommendations'].tolist()[0]
            # out_similarities = similarities.where(similarities.element_source_id == element_id).orderBy('dot', ascending=False)
            # out_ids = [row['element_target_id'] for row in out_similarities.collect()]

            
            # for out_id in out_ids[:5]:
            #     entry = (s_df_element.where(s_df_element.element_id == out_id))
            #     dot = out_similarities.where(out_similarities.element_target_id == out_id)
            #     if entry.head() is not None and dot is not None:
            #         print(entry.head().element_id, entry.head().title, dot.head().dot)
            # with open(model_path + "filtering_results.pkl", 'rb') as file:
            #     filtering_results = pickle.load(file)
            # recs = filtering_results[element_id][:num_recs]
            outputs = []
            for target_rec in target_recs[:num_recs]:
                outputs.append({'element_id': target_rec.element_id})
            # for rec in recs: 
            #     outputs.append({'id': int(rec[1]), 'score': rec[0]})
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
                    Session.merge(element_entry)
                    Session.commit()
                elif message_type == 'activity':
                    activity_entry = Activity(element_id=package['content']['element_id'], user_id=package['content']['user_id'], rating=package['content']['rating'], status=package['content']['status'])
                    if package['content']['element_id'] is not None:
                        known_element = Session.query(Element).filter(Element.element_id==package['content']['element_id']).limit(10).all()
                        if known_element is not None and not len(known_element) > 0 :
                            element_entry = Element(element_id=package['content']['element_id'], title="", tags="")
                            Session.add(element_entry)
                    Session.commit()
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
