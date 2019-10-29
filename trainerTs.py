import tensorflow as tf
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import *
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
class trainerTs:
     model = None
     yenc = None
     yinv = None
     allY = None
     def __init__(self):
        pass
     def reload_model(self, model_path):
          self.model = load_model(model_path)
          self.graph = tf.get_default_graph()
          self.allY = ['C1','C2','C3',
                    'C4','C5','C6','C7','C8','C9','C10','C11','C12','C13',
                    'C14','C15','C16','C17','C18','C19','C20','C21','C22',
                    'C23','C24','C25','C26','C27','C28','C29','C30','C31',
                    'C32','C33','C34','C35','C36','C37','C38','C39','C40']
          self.yenc = LabelEncoder()
          self.yenc.fit(self.allY)
          self.yinv = [i for i in range(40)]
          self.yinv = self.yenc.inverse_transform(np.int_(self.yinv))

     def predict(self, mano):
          print(mano)
          pred = None
          with self.graph.as_default():
               pred = self.model.predict(mano)
               v = np.argmax(pred[0])
               pred = self.yinv[v]
               #print('{}: {} {}'.format(v, self.allY[v], pred))
          print('Predicted {}'.format(pred))
          return pred

     def create_model(self, input_data_path, model_output_path):
          self.allY = ['C1','C2','C3',
                    'C4','C5','C6','C7','C8','C9','C10','C11','C12','C13',
                    'C14','C15','C16','C17','C18','C19','C20','C21','C22',
                    'C23','C24','C25','C26','C27','C28','C29','C30','C31',
                    'C32','C33','C34','C35','C36','C37','C38','C39','C40']
          self.yenc = LabelEncoder()
          self.yenc.fit(self.allY)
          self.yinv = [i for i in range(40)]
          self.yinv = self.yenc.inverse_transform(np.int_(self.yinv))
          games_df = pd.read_csv(input_data_path)
          d = games_df.drop(['pid'], axis=1)
          #scale the data so they are in range 0-1 so for colums 'Carta 1', 'Carta 2', 'Carta 3' and 'Giocata Avv' divide it by 40
          # while 'Briscola' is to be divided by 4
          d['Carta 1'] = d['Carta 1']/40.0
          d['Carta 2'] = d['Carta 2']/40.0
          d['Carta 3'] = d['Carta 3']/40.0
          # d['Giocata'] = d['Giocata']/40.0
          d['Giocata Avv'] = d['Giocata Avv']/40.0
          d['Briscola']=d['Briscola']/4
          # d['Giocata'] = str(d['Giocata'])
          X = d.drop(['Giocata','Unnamed: 0'], axis=1)
          
          Y = np_utils.to_categorical(self.yenc.transform(d[['Giocata']].values))          
          print(Y)
          
          self.model = Sequential()
          self.model.add(LSTM(50, input_dim=[5,5], activation='relu'))
          #self.model.add(Dense(50, input_dim = 5, activation = 'relu'))
          self.model.add(Dense(100, activation = 'relu'))
          self.model.add(Dense(50, activation = 'relu'))
          self.model.add(Dense(40, activation = 'linear'))
          self.model.compile(loss = 'mse', optimizer = 'adam')

          self.model.fit(
               X,
               Y,
               epochs=5
          )
          self.model.save(model_output_path)


# md = trainerTs()
# # md.create_model('sl2_games.csv','km2.model')
# md.reload_model('km2.model')
# #14,18,38,3,9, C38
# tp=pd.DataFrame([[14/40.0,18/40.0,38/40.0,3/4.0,9/40.0]])
# pdr = md.predict(tp)
# tp2=pd.DataFrame([[0/40.0,0/40.0,18/40.0,1/4.0,7/40.0]])
# pdr2 = md.predict(tp2)
# #11,17,28,2,35,C28
# tp3=pd.DataFrame([[11/40.0,17/40.0,28/40.0,2/4.0,35/40.0]])
# pdr3 = md.predict(tp3)
# print(pdr)
# print(pdr2)
