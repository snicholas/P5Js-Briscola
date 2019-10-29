import tensorflow as tf
import pandas as pd
import random
import numpy as np
#from tensorflow.keras.models import Dense, Sequential, load_model
#from keras.layers import *

status = ['', 'p1hand', 'p1keep', 'p1played', 'p2keep', 'p2played']
seeds = ['Cuori', 'Quadri', 'Fiori', 'Picche']
game = np.array([0. for x in range(41)])

game[40] = random.choice([0,1,2,3])/4
for i in range(3):
    p = random.choice([x for x in range(40)])
    game[p]=1/6
ch = [i for i in range(40) if game[i]==1]
print(game.shape)
