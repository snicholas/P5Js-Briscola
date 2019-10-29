import random
import pandas as pd
import trainer
from sklearn.preprocessing import OneHotEncoder
import trainerTs
semi = ['Cuori', 'Quadri', 'Fiori', 'Picche']


class player:
    def __init__(self, carte, pid, t,tfm):
        self.carte = carte
        self.punti = 0
        self.pid = pid
        self.t =t
        self.tfm=tfm

    def aggiungiPunti(self, c1, c2):
        self.punti += c1.valore+c2.valore

    def aggiungiCarta(self, carta):
        # print("pescato : {} ".format(carta))
        self.carte.append(carta)

    def giocaCarta(self, briscolaIdx, t, cartaAltro = 0, carta = None):
        casuale=False
        if carta is not None:
            if(carta in self.carte):
                self.carte.remove(carta)
            return carta, False
        if len(self.carte) == 0 :
            return None, None
        mp1=[]
        for c in range(3):
            if len(self.carte)>c:
                mp1.append(self.carte[c].id)
            else:
                mp1.append(self.carte[0].id)
        if self.pid == 'p2':
            c = t.choose([[mp1[0],mp1[1], mp1[2], briscolaIdx, cartaAltro]])
        else:
            dd = pd.DataFrame([[mp1[0]/40.0,mp1[1]/40.0, mp1[2]/40.0, briscolaIdx/4.0, cartaAltro/40.0]])
            c = int(self.tfm.predict(dd)[1:])
        if c is None or c not in mp1:
            if self.pid == 'p1':
                casuale=True
            c = random.choice([mp1[0],mp1[1], mp1[2]])
        c = self.carte[[x.id for x in self.carte].index(c)]
        if(c in self.carte):
            self.carte.remove(c)
        return c, casuale
    
class carta:
    def __init__(self, numero, seme):
        self.numero = numero
        self.seme = seme
        self.id = numero+(10*semi.index(seme))
        self.valore = 0
        if self.numero==1:
            self.valore = 11
        elif self.numero == 3:
            self.valore = 10
        elif self.numero == 8:
            self.valore = 2
        elif self.numero == 9:
            self.valore = 3
        elif self.numero == 10:
            self.valore = 4

    def getLabel(self):
        return "C{}".format(self.id)
    def getValue(self):
        return "{}".format(self.id)
        # return "{}{}".format(self.numero,self.seme)
    def __str__(self):
        #"Hello, {}. You are {}.".format(name, age)
        return "{} di {} valore = {}".format(self.numero, self.seme, self.valore)

def chiprende(cartaP1, cartaP2, semeBriscola, pDiTurno):
    if cartaP1.seme == semeBriscola and cartaP2.seme != semeBriscola:
        return 1
    elif cartaP2.seme == semeBriscola and cartaP1.seme != semeBriscola:
        return 2
    else:
        if cartaP1.valore > cartaP2.valore and (cartaP1.seme==cartaP2.seme or pDiTurno == 1):
            return 1
        elif cartaP2.valore > cartaP1.valore and (cartaP1.seme==cartaP2.seme or pDiTurno == 2):
            return 2
        if cartaP1.valore == cartaP2.valore:
            if cartaP1.numero > cartaP2.numero and (cartaP1.seme==cartaP2.seme or pDiTurno == 1) :
                return 1
            elif cartaP2.numero > cartaP1.numero and (cartaP1.seme==cartaP2.seme or pDiTurno == 2):
                return 2
    return pDiTurno

def daiCarta(mazzo):
    c = random.choice(mazzo)
    mazzo.remove(c)
    return c

def sceglicarta(carte, briscola, giocataAvv=None):
    if(len(carte)==0):
        return None
    if giocataAvv:
        print("il tuo avversario gioca:")
        print(giocataAvv)
    print('scegli la carta ({}): '.format(briscola))
    print("(0) {}".format(carte[0]))
    if(len(carte)>1):
        print("(1) {}".format(carte[1]))
    if len(carte)>2:
        print("(2) {}".format(carte[2]))
    c = carte[int(input())]
    print("Hai giocato {}".format(c))
    return c

def gioca(cpu, casuali, totali, mazzo,pid):    
    statodeckp1=[0 for x in range(40)]
    statodeckp2=[0 for x in range(40)]
    
    if cpu == False:
        print("La briscola è {}".format(mazzo[-1]))
    briscola = mazzo[-1].seme
    tts = trainerTs.trainerTs()
    tts.reload_model('assets/km2.model')
    p1 = player([daiCarta(mazzo),daiCarta(mazzo),daiCarta(mazzo)], 'p1',trainer.trainer('train_ref.pkl'), tts)
    p2 = player([daiCarta(mazzo),daiCarta(mazzo),daiCarta(mazzo)], 'p2',trainer.trainer('train_v3.pkl'), tts)
    for card in p1.carte:
        statodeckp1[card.id-1]=1
    for card in p2.carte:
        statodeckp2[card.id-1]=1
    playerDiTurno = 1

    daGiocare = True
    partita = []
    trainCase = []
    trainCase_y = []
    mano = [
        p1.carte[0].id,p1.carte[1].id, p1.carte[2].id, p1.punti,
        p2.carte[0].id,p2.carte[1].id, p2.carte[2].id, p2.punti,
        semi.index(briscola), playerDiTurno
    ]
    if cpu==True:
        c1,cas = p1.giocaCarta(semi.index(briscola),p1.t)
        # if cas:
        #     casuali+=1
    else:
        c1,cas = p1.giocaCarta(semi.index(briscola), p1.t, 0,sceglicarta(p1.carte,briscola))
    #
    c2,cas = p2.giocaCarta(semi.index(briscola),p2.t)
    if cas:
            casuali+=1
    totali+=1
    if cpu == False:
        print("il tuo avversario gioca:")
        print(c2)
    while daGiocare:
        mano.append(c1.id)
        mano.append(c2.id)
        mano.append(playerDiTurno)
        tp = playerDiTurno 
        playerDiTurno = chiprende(c1,c2,briscola, playerDiTurno)
        mp1=[]
        mp2=[]
        for c in range(2):
            if len(p1.carte)>c:
                mp1.append(p1.carte[c].getValue())
            else:
                mp1.append(0)
        
        for c in range(2):
            if len(p2.carte)>c:
                mp2.append(p2.carte[c].getValue())
            else:
                mp2.append(0)
        if cpu == True:
            rf = 0
            if playerDiTurno==1 and (c1.valore>0 or c2.valore>0) :
                if tp == 2:
                    rf = c2.getValue()
                partita.append([pid, mp1[0],mp1[1], c1.getValue(), semi.index(briscola),rf,c1.getLabel()])                
            elif playerDiTurno==2 and (c2.valore>0 or c1.valore>0) :
                if tp == 1:
                    rf = c1.getValue()
                partita.append([pid, mp2[0],mp2[1], c2.getValue(), semi.index(briscola),rf,c2.getLabel()])
        mano.append(playerDiTurno)
        #partita.append(mano)
        if len(mazzo)>0:
            if playerDiTurno == 1:
                p1.aggiungiPunti(c1,c2)
                p1.aggiungiCarta(daiCarta(mazzo))
                p2.aggiungiCarta(daiCarta(mazzo))        
            else:
                p2.aggiungiPunti(c1,c2)
                p2.aggiungiCarta(daiCarta(mazzo))
                p1.aggiungiCarta(daiCarta(mazzo))
        else:
            if playerDiTurno == 1:
                p1.aggiungiPunti(c1,c2)
            else:
                p2.aggiungiPunti(c1,c2)
        mano = []
        for c in range(2):
            if len(p1.carte)>c:
                mano.append(p1.carte[c].id)
            else:
                mano.append(0)
        mano.append(p1.punti)

        for c in range(2):
            if len(p2.carte)>c:
                mano.append(p2.carte[c].id)
            else:
                mano.append(0)
        mano.append(p2.punti)
        mano.append(semi.index(briscola))
        if cpu == False:
            print("-----------------------")
            print("Giocata p1: {}".format(c1))
            print("Giocata p2: {}".format(c2))
            print("-----------------------")
            print('Il giocatore p{} prende {} punti'.format(playerDiTurno,c1.valore+c2.valore))
            print("Punteggio attuale p1:{} p2:{}".format(p1.punti, p2.punti))
            print("-----------------------")
        #c1 = p1.giocaCarta(semi.index(briscola))
        if playerDiTurno == 1:
            if cpu==True:
                c1,cas = p1.giocaCarta(semi.index(briscola), p1.t)
                # if cas:
                #     casuali+=1
            else:
                c1,cas = p1.giocaCarta(semi.index(briscola), p1.t, 0,sceglicarta(p1.carte,briscola))
            c2,cas = p2.giocaCarta(semi.index(briscola), p2.t)
            if cas:
                casuali+=1
            if cpu == False and c2 !=None:
                print("il tuo avversario gioca:")
                print(c2)
        else:
            c2,cas = p2.giocaCarta(semi.index(briscola), p2.t)
            if c2!=None:
                if cas:
                    casuali+=1
                if cpu==True:
                    c1,cas = p1.giocaCarta(semi.index(briscola), p1.t)
                    # if cas:
                    #     casuali+=1
                else:
                    c1,cas = p1.giocaCarta(semi.index(briscola), p1.t, c2.id,sceglicarta(p1.carte,briscola, c2))
        if c1 == None or c2 == None:
            #print("mazzo {} carte p1 {} carte p2 {}".format(len(mazzo), len(p1.carte),len(p2.carte)))
            daGiocare = False
        else:
            totali+=1
        
    # p2.t.train(trainCase,trainCase_y)
    #print("{} a {}".format(p1.punti, p2.punti))
    #print(partita)
    # df = pd.DataFrame(partita)
    # df.to_csv('sl_games.csv')
    vincitore = 1 if p1.punti>p2.punti else 2
    if cpu == False:
        print("vincitore p{} {} a {}".format(vincitore,p1.punti, p2.punti))
    #else:
        #p1.t.saveTraining()
        #p1.t.loadTraining()
        #p2.t.saveTraining()
    return vincitore, casuali,totali, partita

partite = []
casuali = 0
totali = 0
print("inizio")
games = None
cols = ['pid','Carta 1','Carta 2','Carta 3','Briscola', 'Giocata Avv', 'Giocata']

pid=0
mazzo = []
for seme in semi:
    for i in range(1,11):
        c = carta(i,seme)
        mazzo.append(c)
for j in range(1):
    cs=0
    to=0    
    random.shuffle(mazzo)
    for i in range(2):
        # print("partita {}".format(pid))        
        res,cs,to,game=gioca(True, cs,to, mazzo.copy(),pid)
        pid+=1
        partite.append(res)
        g=pd.DataFrame(game)
        g.columns = cols
        if games is None:
            # print("games is none!!!")
            games=g.copy()
        else:
            games = pd.concat([games,g])
    #print("vincitore p{}".format(res))
    to=to/2
    print("parziale {} - vinte {} su {} {:.2f}%".format(j, partite.count(1), len(partite), 100 - 100*(len(partite)-partite.count(1))/len(partite)))
    percCs = 100 - 100*(to-cs)/to
    print("selezioni casuali (parziale) {}/{} {:.2f}%".format(cs,to, percCs))
    casuali+=cs
    totali+=to
# if games is not None:
#     games.reset_index(drop=True, inplace=True)
#     games.to_csv('sl2_games.csv')

# print("vinte {} su {} {:.2f}%".format(partite.count(1), len(partite), 100 - 100*(len(partite)-partite.count(1))/len(partite)))
# percCasuali = 100 - 100*(totali-casuali)/totali
# print("selezioni casuali {}/{} {:.2f}%".format(casuali,totali, percCasuali))
# cs=0
# to=0
# mazzo = []
# for seme in semi:
#     for i in range(1,11):
#         c = carta(i,seme)
#         print(c.getLabel())
#         mazzo.append(c)
# random.shuffle(mazzo)
# res = gioca(False,cs,to,mazzo)
# print("vincitore p{}".format(res))