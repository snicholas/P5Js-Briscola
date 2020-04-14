from flask import Flask,jsonify
from flask import request
from flask import render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from trainerRL import Model
app = Flask(__name__)
cors = CORS(app, resources={r"/briscola/*": {"origins": "*"}})

def init():
    global model, model2
    model=Model(3)
    model.load_weights('assets/rl500_2_128chuby_3')
    model2=Model(3)
    model2.load_weights('assets/rl500_2_128chuby_2')


@app.route("/play",  methods=['GET'])
def game():
    return render_template('index.html', name="home")

@app.route("/briscola/playcard", methods=['POST'])
def playcard():
    inData =  request.get_json(force=True)
    deck,seed,player=inData["deck"],inData['seed'], inData['player']
    deck=[statusesRl.index(x)/6. for x in deck ]
    deck.append(seed/4.)
    deck=np.array(deck)
    if player==1:
        action, _  = model.action_value(deck[None, :])
    else:    
        action, _  = model2.action_value(deck[None, :])
    print('action', action)
    return jsonify(int(action))

statusesRl = ['indeck', 'hand', 'played', 'p2played','p1taken', 'p2taken']

init()

if __name__ == '__main__':
    app.run(debug=True)
    init()
    
    app.run(debug=True)
