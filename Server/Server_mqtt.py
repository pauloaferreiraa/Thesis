#!python3
#!/usr/bin/env python3


import paho.mqtt.client as mqtt
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import warnings, json
warnings.filterwarnings('ignore')
import traceback, time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import mode
from scipy.fftpack import fft
import scipy.fftpack

from flask import Flask
from flask import request

import linecache
import sys
import traceback, pdb


app = Flask(__name__)


classes_meaning= {1:'still_hands_back',2:'still_hands_side'}
    



def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))


freq = '9090U' # ~19ms, using sampling frequency is 52samples/sec
def read_data():
    print('::::Read Data::::')
       
    dt = datetime.now()
    data = pd.DataFrame()
    data_files = []
        
    #for f_id in range(1,16):
    f = '../NewDataSet/Dataset.csv'
    # f = '../Teste_dataset.csv'
    data = pd.read_csv(f, 
                        names=['index','xL', 'yL', 'zL', 'xR', 'yR', 'zR', 'xC', 'yC', 'zC', 'label'], 
                        header=None, index_col=0)

    # print('Data.shape________ %f' % data.shape[0])
    data.index = pd.date_range(start='00:00:00', periods=data.shape[0], freq=freq)
    
    # Filter and clean data
    data = data.dropna()
    data = data[data['label'] != 0]   # some rows are misclassified as 0
    
        
    return data

def rms(ts): 
    return np.sqrt(np.mean(ts**2))

def fft_zero(ts): 
    return abs(fft(ts, axis=0)[0])

def fft_positive(ts): 
    return abs(fft(ts, axis=0)[1])

def fft_negative(ts): 
    return abs(fft(ts, axis=0)[2])

def corrL(df): 
    cor = df.corr()
    return pd.DataFrame({'xyL':[cor['xL']['yL']], 'xzL':[cor['xL']['zL']], 'yzL':[cor['yL']['zL']]})

def corrR(df): 
    cor = df.corr()
    return pd.DataFrame({'xyR':[cor['xR']['yR']], 'xzR':[cor['xR']['zR']], 'yzR':[cor['yR']['zR']]})


def corrC(df): 
    cor = df.corr()
    return pd.DataFrame({'xyC':[cor['xC']['yC']], 'xzC':[cor['xC']['zC']], 'yzC':[cor['yC']['zC']]})


def get_advanced_features(data, y, wsize_sec, overlap=.5):
    print('::::START:::: Get Advance Features ::::')
    
    wsize = int(10*wsize_sec)

    feats = data[['xL','yL','zL']].rolling(wsize,int(wsize/2)).mean().add_suffix('_mean_l')
    
    feat = data[['xR','yR','zR']].rolling(wsize,int(wsize/2)).mean().add_suffix('_mean_r')
    feats = feats.join(feat)
    feat = data[['xC','yC','zC']].rolling(wsize,int(wsize/2)).mean().add_suffix('_mean_c')
    feats = feats.join(feat)

    feat = data[['xR','yR','zR']].rolling(wsize, int(wsize/2)).std().add_suffix('_std_r')
    feats = feats.join(feat)
    feat = data[['xL','yL','zL']].rolling(wsize, int(wsize/2)).std().add_suffix('_std_l')
    feats = feats.join(feat)
    feat = data[['xC','yC','zC']].rolling(wsize, int(wsize/2)).std().add_suffix('_std_c')
    feats = feats.join(feat)
    
    feat = data[['xL','yL','zL']].rolling(wsize, int(wsize/2)).var().add_suffix('_var_l')
    feats = feats.join(feat)
    feat = data[['xR','yR','zR']].rolling(wsize, int(wsize/2)).var().add_suffix('_var_r')
    feats = feats.join(feat)
    feat = data[['xC','yC','zC']].rolling(wsize, int(wsize/2)).var().add_suffix('_var_c')
    feats = feats.join(feat)
    
    feat = data[['xL','yL','zL']].rolling(wsize, int(wsize/2)).apply(rms).add_suffix('_rms_l')
    feats = feats.join(feat)
    feat = data[['xR','yR','zR']].rolling(wsize, int(wsize/2)).apply(rms).add_suffix('_rms_r')
    feats = feats.join(feat)
    feat = data[['xC','yC','zC']].rolling(wsize, int(wsize/2)).apply(rms).add_suffix('_rms_c')
    feats = feats.join(feat)

    #Fast Fourier Transform

    feat = data[['xC','yC','zC']].rolling(wsize, int(wsize/2)).apply(fft_zero).add_suffix('_fft_c_zero')
    feats = feats.join(feat)

    feat = data[['xL','yL','zL']].rolling(wsize, int(wsize/2)).apply(fft_zero).add_suffix('_fft_l_zero')
    feats = feats.join(feat)

    feat = data[['xR','yR','zR']].rolling(wsize, int(wsize/2)).apply(fft_zero).add_suffix('_fft_r_zero')
    feats = feats.join(feat)

    feat = data[['xC','yC','zC']].rolling(wsize, int(wsize/2)).apply(fft_positive).add_suffix('_fft_c_positive')
    feats = feats.join(feat)

    feat = data[['xL','yL','zL']].rolling(wsize, int(wsize/2)).apply(fft_positive).add_suffix('_fft_l_positive')
    feats = feats.join(feat)

    feat = data[['xR','yR','zR']].rolling(wsize, int(wsize/2)).apply(fft_positive).add_suffix('_fft_r_positive')
    feats = feats.join(feat)

    feat = data[['xC','yC','zC']].rolling(wsize, int(wsize/2)).apply(fft_negative).add_suffix('_fft_c_negative')
    feats = feats.join(feat)

    feat = data[['xL','yL','zL']].rolling(wsize, int(wsize/2)).apply(fft_negative).add_suffix('_fft_l_negative')
    feats = feats.join(feat)

    feat = data[['xR','yR','zR']].rolling(wsize, int(wsize/2)).apply(fft_negative).add_suffix('_fft_r_negative')
    feats = feats.join(feat)

        
    # mean_mag = (data**2).sum(axis=1).rolling(wsize, int(wsize/2)).apply(lambda ts: np.sqrt(ts).mean())
    # mean_mag.name = 'mean_mag'
    # feats = feats.join(mean_mag) 
    
    pairs_cor_ = data[['xL','yL','zL']].rolling(window=int(wsize/2)).corr(other=data[['xL','yL','zL']])
    feats = feats.join(pairs_cor_)
    pairs_cor_ = data[['xR','yR','zR']].rolling(window=int(wsize/2)).corr(other=data[['xR','yR','zR']])
    feats = feats.join(pairs_cor_)
    pairs_cor_ = data[['xC','yC','zC']].rolling(window=int(wsize/2)).corr(other=data[['xC','yC','zC']])
    feats = feats.join(pairs_cor_)
    
    # y = data[['label']].rolling(wsize, int(wsize/2)).apply(lambda ts: mode(ts)[0])  
    
    feats = feats.iloc[int(wsize*overlap)::int(wsize*overlap)] 
    

    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats = feats.fillna(0)

    y = y.iloc[int(wsize*overlap)::int(wsize*overlap)]
    print('::::END:::: Get features Advance::::')
    
    # print(feats.shape)
    # print(data.shape)
    return feats, y

def train_model(X, y, est, grid):
    print('::::Train Model::::')
    gs = GridSearchCV(estimator=est, param_grid=grid, scoring='accuracy', cv=5, n_jobs=-1)
    gs = gs.fit(X, y.values.ravel())
    
    return (gs.best_estimator_, gs.best_params_)

def eval_model(mod, X_test, y_test, mod_name, plt_roc=True):
    print('::::Eval Model::::')
    y_prob = mod.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=[1,2,3,4,5,6])
    
    y_test_bin_ravel = y_test_bin.ravel()
    y_prob_ravel = y_prob.ravel()
    
    y_test_bin_ravel = y_test_bin_ravel[:len(y_prob_ravel)]
    

    fpr, tpr, _ = roc_curve(y_test_bin_ravel, y_prob_ravel)
    roc_auc = auc(fpr, tpr)
    
    # if plt_roc:
    #     plt.plot(fpr, tpr, lw=2,
    #              label='average ROC curve (auc=%0.2f), model: %s' % (roc_auc,mod_name))
    #     plt.legend(loc="lower right")

    y_pred = cross_val_predict(mod,X_test,y_test,cv=20)
    score = accuracy_score(y_true=y_test, y_pred=y_pred)
    print('Accuracy score on the test set: %.3f' %score)

    # cross_validation = cross_val_score(mod,X_test,y_test,cv=20)
    # print('Cross validation %.3f' % cross_validation.mean())
    
    confusion_ma = confusion_matrix(y_true=y_test, y_pred=y_pred)
    confusion_ma = pd.DataFrame(confusion_ma, index=list(range(1,6)), columns=list(range(1,6)))
    print('Confusion Matrix...')
    print(confusion_ma)

    
    return (roc_auc, score)


data = read_data()
param_range = [100]



# feats, y = get_advanced_features(data, 2)

# split data into train and test sets
y = data[['label']]
data = data.drop('label',axis=1)


X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=.25, random_state=0,shuffle=False)

# X_train, X_test, y_train, y_test = train_test_split(feats, y, test_size=.25, random_state=0, stratify=y)


X_train, y_train = get_advanced_features(X_train, y_train, 2)
X_test, y_test = get_advanced_features(X_test, y_test, 2)

# from collections import Counter

# print(Counter(list(y_test['label'].values)))


print('Support Vector Machine')
svm_model, params = train_model(X_train, y_train, 
                    est=SVC(probability=True),
                    grid={'C': param_range, 'gamma': param_range, 'kernel': ['linear']})
roc_auc, acc = eval_model(svm_model, X_test, y_test, 'SVC')
print(params)
print('AUC score: %.3f' % acc)           

print('K-Nearest Neighbor')
knn_model, params = train_model(X_train, y_train, 
                    est=KNeighborsClassifier(),
                    grid={'n_neighbors':[5, 8, 10, 12], 'weights':['uniform', 'distance']})
roc_auc, acc = eval_model(knn_model, X_test, y_test,'KNN')
print('AUC score: %.3f' % acc)

print('Random Forest')
model, params = train_model(X_train, y_train, 
                    est=RandomForestClassifier(n_jobs=-1, criterion='entropy'),
                    grid={'n_estimators':[10,30,100]})
eval_model(model, X_test, y_test,'Forest')

def get_advanced_features_predict(data, wsize_sec, overlap=.5):
    print('::::START:::: Get Advance Features ::::')
    def rms(ts): return np.sqrt(np.mean(ts**2))

    #print(data)
    wsize = int(10*wsize_sec)
    
    
    feats = data[['xL','yL','zL']].rolling(wsize,int(wsize/2)).mean().add_suffix('_mean_l')
    
    feat = data[['xR','yR','zR']].rolling(wsize,int(wsize/2)).mean().add_suffix('_mean_r')
    feats = feats.join(feat)
    feat = data[['xC','yC','zC']].rolling(wsize,int(wsize/2)).mean().add_suffix('_mean_c')
    feats = feats.join(feat)
    feat = data[['xL','yL','zL']].rolling(wsize, int(wsize/2)).std().add_suffix('_std_l')
    feats = feats.join(feat)
    feat = data[['xR','yR','zR']].rolling(wsize, int(wsize/2)).std().add_suffix('_std_r')
    feats = feats.join(feat)
    feat = data[['xC','yC','zC']].rolling(wsize, int(wsize/2)).std().add_suffix('_std_c')
    feats = feats.join(feat)

    
    
    
    feat = data[['xL','yL','zL']].rolling(wsize, int(wsize/2)).var().add_suffix('_var_l')
    feats = feats.join(feat)
    feat = data[['xR','yR','zR']].rolling(wsize, int(wsize/2)).var().add_suffix('_var_r')
    feats = feats.join(feat)
    feat = data[['xC','yC','zC']].rolling(wsize, int(wsize/2)).var().add_suffix('_var_c')
    feats = feats.join(feat)
    
    feat = data[['xL','yL','zL']].rolling(wsize, int(wsize/2)).apply(rms).add_suffix('_rms_l')
    feats = feats.join(feat)
    feat = data[['xR','yR','zR']].rolling(wsize, int(wsize/2)).apply(rms).add_suffix('_rms_r')
    feats = feats.join(feat)
    feat = data[['xC','yC','zC']].rolling(wsize, int(wsize/2)).apply(rms).add_suffix('_rms_c')
    feats = feats.join(feat)

     #Fast Fourier Transform

    feat = data[['xC','yC','zC']].rolling(wsize, int(wsize/2)).apply(fft_zero).add_suffix('_fft_c_zero')
    feats = feats.join(feat)

    feat = data[['xL','yL','zL']].rolling(wsize, int(wsize/2)).apply(fft_zero).add_suffix('_fft_l_zero')
    feats = feats.join(feat)

    feat = data[['xR','yR','zR']].rolling(wsize, int(wsize/2)).apply(fft_zero).add_suffix('_fft_r_zero')
    feats = feats.join(feat)

    feat = data[['xC','yC','zC']].rolling(wsize, int(wsize/2)).apply(fft_positive).add_suffix('_fft_c_positive')
    feats = feats.join(feat)

    feat = data[['xL','yL','zL']].rolling(wsize, int(wsize/2)).apply(fft_positive).add_suffix('_fft_l_positive')
    feats = feats.join(feat)

    feat = data[['xR','yR','zR']].rolling(wsize, int(wsize/2)).apply(fft_positive).add_suffix('_fft_r_positive')
    feats = feats.join(feat)

    feat = data[['xC','yC','zC']].rolling(wsize, int(wsize/2)).apply(fft_negative).add_suffix('_fft_c_negative')
    feats = feats.join(feat)

    feat = data[['xL','yL','zL']].rolling(wsize, int(wsize/2)).apply(fft_negative).add_suffix('_fft_l_negative')
    feats = feats.join(feat)

    feat = data[['xR','yR','zR']].rolling(wsize, int(wsize/2)).apply(fft_negative).add_suffix('_fft_r_negative')
    feats = feats.join(feat)

    
    # mean_mag = (data**2).sum(axis=1).rolling(wsize, int(wsize/2)).apply(lambda ts: np.sqrt(ts).mean())
    # mean_mag.name = 'mean_mag'
    # feats = feats.join(mean_mag) 
    
    pairs_cor_ = data[['xL','yL','zL']].rolling(window=int(wsize/2)).corr(other=data[['xL','yL','zL']])
    feats = feats.join(pairs_cor_)
    pairs_cor_ = data[['xR','yR','zR']].rolling(window=int(wsize/2)).corr(other=data[['xR','yR','zR']])
    feats = feats.join(pairs_cor_)
    pairs_cor_ = data[['xC','yC','zC']].rolling(window=int(wsize/2)).corr(other=data[['xC','yC','zC']])
    feats = feats.join(pairs_cor_)
    

    feats = feats.replace([np.inf, -np.inf], np.nan)

    feats = feats.iloc[int(wsize*overlap)::int(wsize_sec*overlap)] #PORQUE????
    # feats = feats.iloc[int(wsize_sec*overlap)::int(wsize_sec*overlap)]
    feats = feats.fillna(0)
    # print(feats)
    print('::::END:::: Get features Advance Predict::::')
    return feats


def predict_post(data):
    
    
    global freq
    print('.........')
    # print(data)
    data_frame = []
    for d in data:
        row = [d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9]]
        data_frame += [row]
    pd_data_frame = pd.DataFrame(data_frame,columns=['index','xL','yL','zL','xR','yR','zR','xC','yC','zC'])
    pd_data_frame.index = pd.date_range(start='00:00:00', periods=pd_data_frame.shape[0], freq=freq)

    feats = get_advanced_features_predict(pd_data_frame,2)

    return (knn_model.predict(feats), svm_model.predict(feats),knn_model.predict(feats))

sentData = {}

def parse_data():
   

    jData = []

    l = sentData['Left']
    r = sentData['Right']
    c = sentData['Chest']


    
    # print(l)
    # print(r)
    # print(c)

    nR = 0
    nC = 0
    # print(len(l))
    for n in range(0,len(l)):

        if n >= len(r):
            
            break
        if n >= len(c):
            break
        # print(l[n])
        iL = l[n]['index']
        iR = r[n]['index']
        iC = c[n]['index']

        if iL == iR and iL == iC:
            
            xL = l[n]['x']
            yL = l[n]['y']
            zL = l[n]['z']
            xR = r[n]['x']
            yR = r[n]['y']
            zR = r[n]['z']
            xC = c[n]['x']
            yC = c[n]['y']
            zC = c[n]['z']

            jData += [[iL,xL,yL,zL,xR,yR,zR,xC,yC,zC]]
        
    return jData
            
    
queueLeft = []
queueRight = []
queueChest = []

#the callback function
def on_connect(client, userdata, flags, rc):
    print("Connected With Result Code {}".format(rc))
    client.subscribe('Left')
    client.subscribe('Right')
    client.subscribe('Chest')

def on_disconnect(client, userdata, rc):
	print("Disconnected From Broker")


# push to the queue q the data d
def push_queue(q,d):
    global queueLeft
    global queueChest
    global queueRight

    # print('Pushed to queue %s' % q)
    if q == 'Left':
        queueLeft.insert(0,d)
    elif q == 'Right':
        queueRight.insert(0,d)
    else:
        queueChest.insert(0,d)
    
    

def pop_queue(q):
    d = []

    
    if q == 'Left':
        d = queueLeft.pop()
    elif q == 'Right':
        d = queueRight.pop()
    else:
        d = queueChest.pop()

    return d

def on_message(client, userdata, message):
    try:
        # print(message.payload.decode())
        # print(message.topic)
        global sentData
        # print(len(sentData.keys()))
        push_queue(message.topic,json.loads(message.payload.decode()))
        dL = []
        dR = []
        dC = []
        # print('Left queue')
        # print(queueLeft)
        # print('Right queue')
        # print(queueRight)
        # print('Chest queue')
        # print(queueChest)
        # try to pop the 3 queues
        try:
            if len(queueLeft) > 0 and len(queueRight) > 0 and len(queueChest) > 0:
                dL = pop_queue('Left')
                dR = pop_queue('Right')
                dC = pop_queue('Chest')

                if dL and dR and dC:

                    # #need to check whether the first index are equal or not
                    # iL = dL[0]['index']
                    # iR = dR[0]['index']
                    # iC = dC[0]['index']                        

                    sentData['Left'] = dL
                    sentData['Right'] = dR
                    sentData['Chest'] = dC

                    # print('Parse.....')
                    # print(sentData)
                    jData = parse_data()
                    # print(jData)
                    k, s, r = predict_post(jData)
                    client.publish('server',k.mean())
                    print(k)
                    print(s)
                    print(r)
                    
                    sentData = {}
            
        except Exception as e:
            print(e)
            print(traceback.print_exc())

    except Exception as e:
        print(e)
        print(traceback.print_exc())

# broker_address = "iot.eclipse.org"
broker_address = "test.mosquitto.org"
# broker_address = 'broker.hivemq.com'
broker_portno = 1883
client = mqtt.Client()

#Assigning the object attribute to the Callback Function
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_message = on_message

client.connect(broker_address, broker_portno)

client.loop_forever()