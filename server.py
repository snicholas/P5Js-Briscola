from flask import Flask,jsonify
from flask import request
from flask import render_template
from flask_cors import CORS
import tensorflow as tf
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import *
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
app = Flask(__name__)
cors = CORS(app, resources={r"/briscola/*": {"origins": "*"}})

def init():
    global model, graph, yinv
    model = load_model('assets/km2.model')
    graph = tf.get_default_graph()
    allY = ['C1','C2','C3',
                    'C4','C5','C6','C7','C8','C9','C10','C11','C12','C13',
                    'C14','C15','C16','C17','C18','C19','C20','C21','C22',
                    'C23','C24','C25','C26','C27','C28','C29','C30','C31',
                    'C32','C33','C34','C35','C36','C37','C38','C39','C40']
    yenc = LabelEncoder()
    yenc.fit(allY)
    yinv = [i for i in range(40)]
    yinv = yenc.inverse_transform(np.int_(yinv))

@app.route("/play",  methods=['GET'])
def game():
    return render_template('index.html', name="home")

@app.route("/briscola/playcard", methods=['POST'])
def playcard():
    inData =  request.get_json(force=True)
    print(inData)
    c1,c2,c3,briscola,ca = inData['c1'],inData['c2'],inData['c3'],inData['briscola'],inData['ca']
    dd = pd.DataFrame([[c1/40.0,c2/40.0, c3/40.0, briscola/4.0, ca/40.0]])
    pred = None
    with graph.as_default():
        pred = model.predict(dd)
        v = np.argmax(pred[0])
        pred = yinv[v]
    print('Predicted {}'.format(pred))
    
    c = int(pred[1:])
    return jsonify(c)

if __name__ == '__main__':
    #app.run(debug=True)
    init()
    app.run(debug=True)