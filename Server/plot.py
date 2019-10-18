import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime, timedelta



freq = '9090U' # ~19ms, using sampling frequency is 52samples/sec
def read_data():
    print('::::Read Data::::')
       
    dt = datetime.now()
    data = pd.DataFrame()
    data_files = []
        
    #for f_id in range(1,16):
    f = '../NewDataSet/Squat_Diana_1.csv'
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

print(read_data())