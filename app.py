import os
import config
import flask

import pandas as pd

from flask import (Flask, session, g, json, Blueprint,flash, jsonify, redirect, render_template, request,
                   url_for, send_from_directory, send_file)
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel 
from sklearn.neighbors import NearestNeighbors

from scipy.sparse import csr_matrix

app = Flask(__name__)
cors = CORS(app)
app.config.from_object(os.environ['APP_SETTINGS'])


def init_dataset(path, limit=0):
    df = pd.read_csv(path, sep=';', na_filter=True, error_bad_lines=False, names=['id', 'title', 'tags'], skiprows=1)
    
    def transform_tag_string(tags):
        if isinstance(tags, str):
            tags = tags.lower()
            tags = ' '.join(tags.split(','))
            tags = tags.replace('  ', ' ')
            tags = ''.join([x for x in tags if not x.isdigit()])
        return tags

    df['tags'] = df['tags'].apply(lambda x: transform_tag_string(x))
    df = df.dropna(subset = ['id', 'title', 'tags'])

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

tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words=['и', 'или'])
tfidf_matrix = tf.fit_transform(df['tags'].values.astype(str))

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix) 
results = {}
for idx, row in df.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1] 
    similar_items = [(cosine_similarities[idx][i], df['id'][i]) for i in similar_indices] 
    results[row['id']] = similar_items[1:]
print('Similarities found')

model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model_knn.fit(mat_book_features)
print('KNN model created')

def translate_indices_mat_to_df(indices):
    indices=indices[0]
    translated_indices = []
    for idx in indices:
        translated_indices.append(df_book_features.index[idx])
    return translated_indices

def translate_idx_df_to_mat(idx):
    df_indices = df_book_features.index.tolist()
    return df_indices.index(idx)

def item(book_id):  
    return df.loc[df['id'] == book_id]['title'].tolist()[0]

def extract_filtered_recs(book_id, num):
    recs = results[book_id][:num]
    outputs = []
    for rec in recs: 
        outputs.append({'id': int(rec[1]), 'title': item(rec[1]), 'score': rec[0]})
    return outputs

def extract_knn_recs(book_id, num):
    outputs = []
    
    distances, indices = model_knn.kneighbors(
            mat_book_features[translate_idx_df_to_mat(book_id)],
            n_neighbors=10)
    distances = distances[0]
    indices = translate_indices_mat_to_df(indices)
    # print(indices)
    recs = zip(distances, indices)
    # print(recs[0])
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
        return response

@app.route('/api/0.1/knn_recommender', methods=['POST', 'GET'])
@cross_origin()
def knn_recommender():
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
        return response

if __name__ == "__main__":
    app.run(host='0.0.0.0', port='5002', debug=True, threaded=True)