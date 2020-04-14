import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import random
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import gc
import matplotlib
import matplotlib.pyplot as plt
import time
tf.executing_eagerly()


class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        # sample a random categorical action from given logits
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_policy')
        # no tf.get_variable(), just simple Keras API
        self.hidden1 = kl.Dense(128, activation='relu')
        self.hidden2 = kl.Dense(128, activation='relu')
        self.value = kl.Dense(1, name='value')
        # logits are unnormalized log probabilities
        self.logits = kl.Dense(num_actions, name='policy_logits')
        self.dist = ProbabilityDistribution()

    def call(self, inputs):
        # inputs is a numpy array, convert to Tensor
        x = tf.convert_to_tensor(inputs)
        # separate hidden layers from the same input tensor
        hidden_logs = self.hidden1(x)
        hidden_vals = self.hidden2(x)
        return self.logits(hidden_logs), self.value(hidden_vals)

    def action_value(self, obs):
        # executes call() under the hood
        logits, value = self.predict(obs)
        action = self.dist.predict(logits)
        # a simpler option, will become clear later why we don't use it
        # action = tf.random.categorical(logits, 1)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

class envn():
    statuses = ['indeck', 'hand', 'played', 'p2played','p1taken', 'p2taken']
    cardOrder=[]
    actualCard = 0
    deck = []
    deckadv = []
    deckfull = []
    seed = -1
    player=-1
    obs = []
    observation_space = []
    turn = 1
    def __init__(self, player, cardOrder=[]):
        if(cardOrder==[]):
            self.cardOrder = [x for x in range(40)]
            random.shuffle(self.cardOrder)
        else:
            self.cardOrder=cardOrder
        self.reset()
        self.player=player
    def reset(self):
        self.actualCard = 0
        self.deckfull = np.array([0 for x in range(40)])
        self.deck = np.array([0 for x in range(40)])
        self.deckadv = np.array([0 for x in range(40)])
        self.seed=self.cardOrder[-1]%10
        for i in range(3):
            c = self.cardOrder[self.actualCard]
            self.deck[c]=1
            self.deckfull[c]=-1
            self.actualCard+=1
        for i in range(3):
            c = self.cardOrder[self.actualCard]
            self.deckadv[c]=1
            self.deckfull[c]=-1
            self.actualCard+=1
        self.obs=self.getobs()
        self.observation_space = self.getobs()
        self.turn=1
        return self.obs
    def getobs(self):
        ob=[]
        ob.extend(self.deck/6.)
        ob.append(self.seed/4.)
        return np.array(ob)
    cardvalues ={
        1:11,2:0,3:10,4:0,5:0,6:0,7:0,8:2,9:3,10:4
    }
    def handwinner(self, card, cardadv):
        isbriscola=card%10==self.seed
        isbriscolaadv=card%10==self.seed
        valore = self.cardvalues[(card%10)+1]
        valoreadv = self.cardvalues[(cardadv%10)+1]
        punti=valore+valoreadv
        if isbriscola and not isbriscolaadv:
            return True, punti
        if not isbriscola and isbriscolaadv:
            return False, punti
        return valore>valoreadv, punti
    def step(self, action):
        reward = 0
        cards = [i for i,x in enumerate(self.deck) if x==1]
        choice=0
        if action >= len(cards):
            ends= len([x for x in self.deck if x==1 or x==0])==0 and len([x for x in self.deckadv if x==1 or x==0])==0
            return self.obs,reward, ends, None
        if len(cards)>0:
            choice=cards[int(action)]
            self.deck[choice]=2
            win, punti = False,-1
            if self.turn==self.player:
                s = [i for i,x in enumerate(self.deckadv) if x==1]
                if len(s)==0:
                    self.render()
                    ends= len([x for x in self.deck if x==1 or x==0])==0 and len([x for x in self.deckadv if x==1 or x==0])==0
                    return self.obs,reward, ends, None
                else:
                    cardadv = random.sample(s,1)[0]
                    self.deckadv[cardadv]=2
                    win, punti=self.handwinner(choice, cardadv)
            else:
                cardadv = [i for i,x in enumerate(self.deckadv) if x==2][0]
                win, punti=self.handwinner(cardadv,choice)
            if win:
                reward+=punti
                self.turn=self.player
                self.deck[choice]=4
                self.deck[cardadv]=4
                self.deckadv[cardadv]=5
                self.deckadv[choice]=5
                if self.actualCard<39:
                    c = self.cardOrder[self.actualCard]
                    self.deck[c] = 1
                    self.deckfull[c]=-1
                    self.actualCard+=1
                    c = self.cardOrder[self.actualCard]
                    self.deckadv[c]=1
                    self.deckfull[c]=-1
                    self.actualCard+=1
            else:
                reward-=punti
                self.turn=2
                self.deck[choice]=5
                self.deck[cardadv]=5
                self.deckadv[cardadv]=4
                self.deckadv[choice]=4
                if self.actualCard<39:
                    c = self.cardOrder[self.actualCard]
                    self.deckadv[c]=1
                    self.deckfull[c]=-1
                    self.actualCard+=1
                    c = self.cardOrder[self.actualCard]
                    self.deck[c] = 1
                    self.deckfull[c]=-1
                    self.actualCard+=1
                if 1 in self.deckadv:
                    cardadv = [i for i,x in enumerate(self.deckadv) if x==1][0]
                self.deckadv[cardadv]=2

            self.obs=self.getobs()
            self.observation_space = self.obs
            ends= len([x for x in self.deck if x==1 or x==0])==0 and len([x for x in self.deckadv if x==1 or x==0])==0
            return self.obs,reward, ends, None
        self.obs=self.getobs()
        return self.obs,reward, len([x for x in self.deck if x==1 or x==0])==0, None


    def render(self):
        print('turn:',self.turn)
        print('deckfull: ',self.deckfull)
        print('deck: ',self.deck)
        print('deckadv: ',self.deckadv)
        print('*'*20)

class A2CAgent:
    def __init__(self, model):
        self.params = {'value': 0.5, 'entropy': 0.01, 'gamma': 0.99}
        self.model = model
        self.model.compile(
            optimizer=ko.RMSprop(lr=0.0001),
            # define separate losses for policy logits and value estimate
            loss=[self._logits_loss, self._value_loss]
        )
    
    def _value_loss(self, returns, value):
        # value loss is typically MSE between value estimates and returns
        return self.params['value']*kls.mean_squared_error(returns, value)

    def _logits_loss(self, acts_and_advs, logits):
        # a trick to input actions and advantages through same API
        actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
        # sparse categorical CE loss obj that supports sample_weight arg on call()
        # from_logits argument ensures transformation into normalized probabilities
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
        # policy loss is defined by policy gradients, weighted by advantages
        # note: we only calculate the loss on the actions we've actually taken
        actions = tf.cast(actions, tf.int64)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        # entropy loss can be calculated via CE over itself
        entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)
        # here signs are flipped because optimizer minimizes
        return policy_loss - self.params['entropy']*entropy_loss

    def test(self, env, render=False):
        obs, done, ep_reward = env.reset(), False, 0
        i=0
        while not done:
            action, _ = self.model.action_value(obs[None, :])
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            i+=1
            if i>=200:
                done=True
            if render:
                env.render()
                print("reward: %d" % ep_reward)
        return ep_reward
    
    
    def train(self, env, batch_sz=40, updates=1000):
        # storage helpers for a single batch of data
        actions = np.empty((batch_sz,), dtype=np.float)
        rewards, dones, values = np.empty((3, batch_sz))
        observations = np.empty((batch_sz,) + env.observation_space.shape)
        # training loop: collect samples, send to optimizer, repeat updates times
        ep_rews = [0.0]
        next_obs = env.reset()
        for update in range(updates):
            # if update%500==0 and update>0:
            #     print("reward after %d updates: %f" % (update, sum(ep_rews)/(len(ep_rews))))
                
            for step in range(batch_sz):
                observations[step] = next_obs.copy()
                actions[step], values[step] = self.model.action_value(next_obs[None, :])
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])

                ep_rews[-1] += rewards[step]
                if dones[step]:
                    ep_rews.append(0.0)
                    next_obs = env.reset()

            _, next_value = self.model.action_value(next_obs[None, :])
            returns, advs = self._returns_advantages(rewards, dones, values, next_value)
            # a trick to input actions and advantages through same API
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            # performs a full training step on the collected batch
            # note: no need to mess around with gradients, Keras API handles it
            losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
        return ep_rews

    def _returns_advantages(self, rewards, dones, values, next_value):
        # next_value is the bootstrap value estimate of a future state (the critic)
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # returns are calculated as discounted sum of future rewards
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.params['gamma'] * returns[t+1] * (1-dones[t])
        returns = returns[:-1]
        # advantages are returns - baseline, value estimates in our case
        advantages = returns - values
        return returns, advantages

if __name__ == '__main__':
    rh=[]
    negRewDeck=[]
    showChart = False
    tt = [
        [28, 38, 8, 7, 15, 22, 27, 33, 0, 12, 17, 5, 21, 20, 16, 3, 24, 10, 37, 9, 36, 31, 39, 1, 14, 2, 11, 19, 29, 30, 35, 32, 6, 18, 26, 34, 25, 23, 4, 13],
        [23, 0, 13, 2, 7, 4, 9, 27, 24, 26, 6, 17, 31, 12, 37, 36, 18, 28, 33, 8, 38, 21, 35, 29, 34, 11, 5, 30, 14, 10, 16, 22, 39, 20, 1, 25, 19, 15, 3, 32]
    ]
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # tf.debugging.set_log_device_placement(True)
    for i in range(1):
        with tf.Graph().as_default() as graph:
            with tf.device('/CPU:0'):
                print("start", time.process_time())
                print("round ", i+1)
                model=Model(3)
                model.load_weights('assets/rl500_2_128chuby_3')
                env =envn(1)
                print(env.cardOrder)
                agent = A2CAgent(model)
                rewards_history = agent.train(env, updates=5000)
                rewmean=1.*sum(rewards_history)/len(rewards_history)
                if rewmean < 0:
                    negRewDeck.append(env.cardOrder)
                print("Finished training, mean reward:", rewmean)
                rt=[]
                fig, ax = plt.subplots()
                x = [i for i in range(len(rewards_history))] 
                mrt = [sum(rewards_history[:i])/i for i in range(1,len(rewards_history))]
                ax.scatter(x, rewards_history,color='C1', s=2)
                ax.plot(x[1:],mrt, label='Mean', linestyle='--',color='C2')
                ax.set(xlabel='Episode', ylabel='Reward',
                title='Rewards over episodes')
                ax.grid()
                # fig.savefig("test"+str(i)+".png")
                if(showChart):
                    plt.show()
                model.save_weights('assets/rl500_2_128chuby_3')
                model=None
                env=None
                agent=None
                rewards_history=None
                print("end", time.process_time())
                gc.collect()
    if len(negRewDeck)>0:
        # open('decks.txt', 'wb').write(r.content)
        # with open('listfile.txt', 'w+') as filehandle:
        with open('listfile.txt', 'a+') as filehandle:
            for listitem in negRewDeck:
                filehandle.write('%s\n' % listitem)