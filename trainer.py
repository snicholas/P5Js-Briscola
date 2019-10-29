from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pickle
from pathlib import Path
class trainer:
    def __init__(self, pklFIle='defTrain.pkl'):
        self.availableClass = [x for x in range(1,41)]
        self.scaler = StandardScaler()
        self.mlp = None
        self.pklFIle=pklFIle
        self.loadTraining()
        if self.mlp is None:
            print('trainer creato')
            self.mlp = MLPClassifier(
                hidden_layer_sizes=(100,100,100, 100), 
                activation='logistic', 
                max_iter=500
            )
        #print('trainer creato')

    def train(self, xdata, ydata):
        self.scaler.fit(xdata)
        self.mlp.partial_fit(xdata, ydata, classes=self.availableClass)
    
    def choose(self, data):
        try:
            p = self.mlp.predict(data)
            if p[0] not in data[0][0:3]:
                return None
            return p
        except:
            return None

    def saveTraining(self):
        pickle.dump(self.mlp, open(self.pklFIle, 'wb'))
    
    def loadTraining(self):
        if Path(self.pklFIle).is_file():
            tmp = pickle.load(open(self.pklFIle, 'rb'))
            if tmp is not None:
                self.mlp = tmp
