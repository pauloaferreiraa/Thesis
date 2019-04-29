#!/usr/bin/env python3

from flask import Flask
from flask import request

app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello world'

@app.route('/data',methods=['POST'])
def process_data():
    print(':::::::')
    print(request.data)
    print(':::::::')
    return 'Just got data'

if __name__ == '__main__':
    app.run(debug=True)