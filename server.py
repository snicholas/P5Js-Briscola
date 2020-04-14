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
import random 
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

statuses = ['indeck', 'p1hand', 'p1played', 'p2hand', 'p2played',
    'p1taken', 'p2taken'
]

@app.route("/briscola/playcardV2", methods=['POST'])
def playcardV2():
    inData =  request.get_json(force=True)
    print(inData)
    deck, player, seed=inData["deck"],inData["player"], inData["seed"]
    deck=[statuses.index(x) for x in deck ]
    moves = []
    if(player==1):
        deck=[x if x != 3 else 0 for x in deck]
        moves =[x for x in deck if x == 1]
    elif(player==2):
        deck=[x if x != 1 else 0 for x in deck]
        moves =[deck.index(x) for x in deck if x == 3]
    print(deck, moves, seed)
    dd = pd.DataFrame([deck, moves, seed])
    return jsonify(0)

    pred = None
    with graph.as_default():
        pred = model.predict(dd)
        v = np.argmax(pred[0])
        pred = yinv[v]
    print('Predicted {}'.format(pred))
    
    c = int(pred[1:])
    return jsonify(c)

def loss(model, deck, seed, player,action):
    l= 0
    p1played=2 in deck
    p2played=4 in deck
    y_=model([deck,seed,player])
    
    if p1played and p2played:
        c1=deck[deck.index(2)]
        c2=deck[deck.index(4)]
        vp1=getCardValue(c1)
        vp2=getCardValue(c2)
        sp1=int(c1/10)
        sp2=int(c2/10)
        if take(deck, c1, c2, sp1, sp2, vp1, vp2, seed, player):
            loss=1
            if player==1:
                if sp1==seed and sp2!=seed:
                    loss*=0.8
                else:
                    loss*=0.9
                if vp1 > 0 and vp1>vp2:
                    loss*=0.9
                else:
                    loss*=0.8
            else:
                if sp1!=seed and sp2==seed:
                    loss*=0.8
                else:
                    loss*=0.9
                if vp2 >0 and vp1>vp2:
                    loss*=0.9
                else:
                    loss*=0.8
        
        return loss

    
# @app.route("/briscola/train", methods=['POST'])
def playgame():
    
    model = Sequential()
    model.add(Dense(32, input_shape=(40,1,1)),activation='relu')
    model.output(len(40), activation='sigmoid')


    model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
    
    deck=[0 for x in range(40)]
    seed=random.randint(0,3)
    cards = random.sample(range(40), 6)
    for i in range(3):
        deck[cards[i]]=1
    for i in range(3,6):
        deck[cards[i]]=3
    moreToPlay=True
    totp1=0
    totp2=0
    turn='p1'
    while moreToPlay:
        p1 = random.sample([i for i,x in enumerate(deck) if x==1],1)[0]
        p2 = random.sample([i for i,x in enumerate(deck) if x==3],1)[0]
        vp1=getCardValue(p1)
        vp2=getCardValue(p2)
        sp1=int(p1/10)
        sp2=int(p2/10)
        res=0
        if turn=='p1':
            if sp1!=sp2 and (sp1==seed or sp2!=seed):
                totp1+=vp1+vp2
                deck[p1]=5
                deck[p2]=5
                turn='p1'
            elif sp1==sp2 and vp1>vp2:
                totp1+=vp1+vp2
                deck[p1]=5
                deck[p2]=5
                turn='p1'
            else:
                totp2+=vp1+vp2
                deck[p1]=6
                deck[p2]=6
                turn='p2'
        else:
            if sp1!=sp2 and (sp2==seed or sp1!=seed):
                totp2+=vp1+vp2
                deck[p1]=6
                deck[p2]=6
                turn='p2'
            elif sp1==sp2 and vp1>vp2:
                totp2+=vp1+vp2
                deck[p1]=6
                deck[p2]=6
                turn='p2'
            else:
                totp1+=vp1+vp2
                deck[p1]=5
                deck[p2]=5
                turn='p1'
        c=[i for i,x in enumerate(deck) if x==0]
        if len([i for i,x in enumerate(deck) if x==0])>2:
            c = random.sample([i for i,x in enumerate(deck) if x==0], 2)
        if len(c)>0:
            if(turn=='p1'):
                deck[c[0]]=1
                deck[c[1]]=3
            else:
                deck[c[0]]=3
                deck[c[1]]=1
        moreToPlay=len([x for x in deck if x in [0,1,3]])>0
    print (deck) 
    print(totp1, totp2)
    print(len([x for x in deck if x==5]), len([x for x in deck if x==6]))
        
def take(deck, c1, c2, sp1, sp2, vp1, vp2, seed, turn):
    if turn=='p1':
        if sp1!=sp2 and (sp1==seed or sp2!=seed):
            return True
        elif sp1==sp2 and vp1>vp2:
            return True
        else:
            return False
    else:
        if sp1!=sp2 and (sp2==seed or sp1!=seed):
            return True
        elif sp1==sp2 and vp1>vp2:
            return True
        else:
            return False
    return False
def getCardValue(i):
    i= 1 + i%10
    value=0
    if (i == 1) :
        value = 11
    elif (i == 3):
        value = 10
    elif (i == 8):
        value = 2
    elif (i == 9):
        value = 3
    elif (i == 10):
        value = 4
    return value

if __name__ == '__main__':
    #app.run(debug=True)
    # init()
    # app.run(debug=True)
    playgame()