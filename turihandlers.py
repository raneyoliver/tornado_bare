#!/usr/bin/python

from pymongo import MongoClient
import tornado.web

from tornado.web import HTTPError
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options

from basehandler import BaseHandler

import turicreate as tc
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from joblib import dump, load

import pickle
from bson.binary import Binary
import json
import numpy as np

models_dict = {}
game_dsid = 70

class PrintHandlers(BaseHandler):
    def get(self):
        '''Write out to screen the handlers used
        This is a nice debugging example!
        '''
        self.set_header("Content-Type", "application/json")
        self.write(self.application.handlers_string.replace('),','),\n'))

class UploadLabeledDatapointHandler(BaseHandler):
    def post(self):
        '''Save data point and class label to database
        '''
        data = json.loads(self.request.body.decode("utf-8"))

        vals = data['feature']
        fvals = [float(val) for val in vals]
        print(f"length of feature vector: {len(fvals)}")
        label = data['label']
        sess = game_dsid

        dbid = self.db.labeledinstances.insert_one(
            {"feature":fvals,"label":label,"dsid":sess}
            );
        self.write_json({"id":str(dbid),
            "feature":[str(len(fvals))+" Points Received",
                    "min of: " +str(min(fvals)),
                    "max of: " +str(max(fvals))],
            "label":label})

class RequestNewDatasetId(BaseHandler):
    def get(self):
        '''Get a new dataset ID for building a new dataset
        '''
        a = self.db.labeledinstances.find_one(sort=[("dsid", -1)])
        if a == None:
            newSessionId = 1
        else:
            newSessionId = float(a['dsid'])+1
        self.write_json({"dsid":newSessionId})

class GetLargestDatasetId(BaseHandler):
    def get(self):
        '''Get largest ID from all available datasets
        '''
        a = self.db.labeledinstances.find_one(sort=[("dsid", -1)])
        if a == None:
            largestId = 1
        else:
            largestId = float(a['dsid'])
        self.write_json({"dsid":largestId})

class UpdateModelForDatasetIdTuri(BaseHandler):
    def get(self):
        '''Train a new model (or update) for given dataset ID
        '''
        dsid = self.get_int_arg("dsid",default=0)

        data = self.get_features_and_labels_as_SFrame(dsid)

        # fit the model to the data
        acc = -1
        best_model = 'unknown'
        if len(data)>0:
            
            model = tc.classifier.create(data,target='target',verbose=0)# training
            yhat = model.predict(data)
            models_dict[dsid] = model
            self.clf = models_dict[dsid]
            acc = sum(yhat==data['target'])/float(len(data))
            # save model for use later, if desired
            model.save('../models/turi_model_dsid%d'%(dsid))
            

        # send back the resubstitution accuracy
        # if training takes a while, we are blocking tornado!! No!!
        self.write_json({"resubAccuracy":acc})

    def get_features_and_labels_as_SFrame(self, dsid):
        # create feature vectors from database
        features=[]
        labels=[]
        for a in self.db.labeledinstances.find({"dsid":dsid}): 
            features.append([float(val) for val in a['feature']])
            labels.append(a['label'])

        # convert to dictionary for tc
        data = {'target':labels, 'sequence':np.array(features)}

        # send back the SFrame of the data
        return tc.SFrame(data=data)

class PredictOneFromDatasetIdTuri(BaseHandler):
    def post(self):
        '''Predict the class of a sent feature vector
        '''
        data = json.loads(self.request.body.decode("utf-8"))    
        fvals = self.get_features_as_SFrame(data['feature'])
        dsid = 0

        # load the model from the database (using pickle)
        # we are blocking tornado!! no!!
        if dsid not in models_dict.keys():
            print('Loading Model From file')
            models_dict[dsid] = tc.load_model('../models/turi_model_dsid%d'%(dsid))

        self.clf = models_dict[dsid]

        predLabel = self.clf.predict(fvals);
        self.write_json({"prediction":str(predLabel)})

    def get_features_as_SFrame(self, vals):
        # create feature vectors from array input
        # convert to dictionary of arrays for tc

        tmp = [float(val) for val in vals]
        tmp = np.array(tmp)
        tmp = tmp.reshape((1,-1))
        data = {'sequence':tmp}

        # send back the SFrame of the data
        return tc.SFrame(data=data)

# TODO: test out the sklearn dataset responding 
class UpdateModelForDatasetIdSklearn(BaseHandler):
    def get(self):
        '''Train a new model (or update) for given dataset ID
        '''
        dsid = game_dsid
        # create feature vectors and labels from database
        features = []
        labels   = []
        for a in self.db.labeledinstances.find({"dsid":dsid}): 
            features.append([float(val) for val in a['feature']])
            labels.append(a['label'])

        # fit the model to the data
        knn_model = KNeighborsClassifier(n_neighbors=1);
        acc = -1;
        if labels:
            knn_model.fit(features,labels) # training
            lstar = knn_model.predict(features)
            models_dict[dsid] = knn_model
            self.clf = models_dict[dsid]
            acc = sum(lstar==labels)/float(len(labels))

            # just write this to model files directory
            dump(knn_model, '../models/sklearn_model_dsid%d.joblib'%(dsid))

        # fit the model to the data
        dsid += 1
        xgb = XGBClassifier()
        acc_xgb = -1
        if labels:
            xgb.fit(features,labels) # training
            lstar = knn_model.predict(features)
            models_dict[dsid] = xgb
            self.clf = models_dict[dsid]
            acc_xgb = sum(lstar==labels)/float(len(labels))

            # just write this to model files directory
            dump(xgb, '../models/sklearn_model_dsid%d.joblib'%(dsid))


        # send back the resubstitution accuracy
        # if training takes a while, we are blocking tornado!! No!!
        self.write_json({
            "resubAccuracy_knn":acc,
            "resubAccuracy_xgb":acc_xgb})


class PredictOneFromDatasetIdSklearn(BaseHandler):
    def post(self):
        '''Predict the class of a sent feature vector
        '''
        data = json.loads(self.request.body.decode("utf-8"))    

        vals = data['feature'];
        fvals = [float(val) for val in vals];
        fvals = np.array(fvals).reshape(1, -1)
        dsid  = game_dsid
        if not self.use_knn: dsid += 1   # +1 dsid is xgb

        # load the model (using pickle)
        if dsid not in models_dict.keys():
            # load from file if needed
            print('Loading Model From DB')
            try:
                model = load('../models/sklearn_model_dsid%d.joblib'%(dsid)) 
                #model = pickle.loads(tmp['model'])
                models_dict[dsid] = model

            except:
                self.write_json({"prediction":f"No Model for DSID {dsid}"})
                return
            
            
        self.clf = models_dict[dsid]
        print("Current sequence shape:", fvals.shape)
        predLabel = self.predict_move(fvals)
        self.write_json({"prediction":str(predLabel)})

    def predict_move(self, current_sequence):
        dsid  = game_dsid

        # load the model (using pickle)
        if dsid not in models_dict.keys():
            # load from file if needed
            print('Loading Model From DB')
            try:
                model = load('../models/sklearn_model_dsid%d.joblib'%(dsid)) 
                #model = pickle.loads(tmp['model'])
                models_dict[dsid] = model

            except:
                self.write_json({"prediction":f"No Model for DSID {dsid}"})
                return
        else:
            model = models_dict[dsid]
            
        last_move_index = np.max(np.where(current_sequence[0] != 0)[0]) + 1 if np.any(current_sequence[0] != 0) else 0
        print(f"last_move_index: {last_move_index}, current_sequence.shape[1]: {current_sequence.shape[1]}")
        if last_move_index < current_sequence.shape[1]:
            # The sequence has room for more moves
            left_option = current_sequence.copy()
            right_option = current_sequence.copy()

            left_option[0, last_move_index] = 3.0  # Enemy moves left
            right_option[0, last_move_index] = 4.0 # Enemy moves right
        else:
            # The sequence is full, can't add more moves
            # Handle this case as per your game logic (e.g., end the game)
            return None
        
        # Reshape the arrays to 2D for scikit-learn
        left_option = left_option.reshape(1, -1)
        right_option = right_option.reshape(1, -1)

        # Debugging: Print the shapes
        print("Left option shape:", left_option.shape)
        print("Right option shape:", right_option.shape)

        # Predict the outcomes
        left_prediction = model.predict(left_option)
        right_prediction = model.predict(right_option)

        print(f"left: {left_prediction}, right: {right_prediction}")
        # Choose the move with a better outcome
        return "Left" if left_prediction > right_prediction else "Right"

class ChangeModel(BaseHandler):
    def post(self):
        '''
        Change KNN <-> XGB model
        '''
        data = json.loads(self.request.body.decode("utf-8"))
        curr_model = self.use_knn

        model = data['model']
        if model == "KNN":
            self.use_knn = True
        elif model == "XGB":
            self.use_knn = False

        self.write_json({
            "from": f"{'KNN' if curr_model else 'XGB'}",
            "to": f"{'KNN' if self.use_knn else 'XGB'}",
            })

