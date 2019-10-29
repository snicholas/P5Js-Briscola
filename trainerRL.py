import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import random
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

tf.executing_eagerly()


class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        # sample a random categorical action from given logits
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_policy')
        # no tf.get_variable(), just simple Keras API
        self.hidden1 = kl.Dense(16, activation='relu')
        self.hidden2 = kl.Dense(16, activation='relu')
        self.value = kl.Dense(1, name='value')
        # logits are unnormalized log probabilities
        self.logits = kl.Dense(num_actions, name='policy_logits')
        self.dist = ProbabilityDistribution()

    def call(self, inputs):
        # inputs is a numpy array, convert to Tensor
        x = tf.convert_to_tensor(inputs, dtype=tf.int32)
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
    deck = []
    seed = -1
    player=-1
    obs = []
    observation_space = []
    def __init__(self, player):
        self.reset()
        self.player=player
    def reset(self):
        self.deck = np.array([0 for x in range(40)])
        self.seed=random.randint(0,3)
        cards = random.sample(range(40), 6)
        for i in range(3):
            self.deck[cards[i]]=1
        # for i in range(3,6):
        #     self.deck[cards[i]]=3
        self.obs=self.getobs()
        self.observation_space = self.getobs()
        # print(self.obs)
        return self.obs
    def getobs(self):
        ob=[]
        ob.extend(self.deck)
        # ob.append(self.seed)
        samples = [i for i,x in enumerate(self.deck) if x==1]
        for i in range(3-len(samples)):
            samples.append(-1)
        # ob.extend(samples)
        # print(ob)
        return np.array(ob)
    def cardgood(self, action):
        if self.deck[action]==self.player:
            return action
        else:
            samples = [i for i,x in enumerate(self.deck) if x==1]
            return random.sample(samples,1)[0]
    def step(self, action):
        # print('action:',action)
        reward = -1
        if self.deck[action]==self.player:
            self.deck[action]=3
            samples = [i for i,x in enumerate(self.deck) if x==0]
            if(len(samples)>0):
                card = random.sample(samples,1)[0]
                self.deck[card]=self.player
            reward=1
            self.obs=self.getobs()
            self.observation_space = self.obs
            return self.obs,reward, len([x for x in self.deck if x==1 or x==0])==0, None
        self.obs=self.getobs()
        # return self.obs,reward, True, None
        return self.obs,reward, len([x for x in self.deck if x==1 or x==0])==0, None
    def render(self):
        print(self.deck)

class A2CAgent:
    def __init__(self, model):
        self.params = {'value': 0.1, 'entropy': 0.01, 'gamma': 0.99}
        self.model = model
        self.model.compile(
            optimizer=ko.RMSprop(lr=0.001),
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
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        # entropy loss can be calculated via CE over itself
        entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)
        # here signs are flipped because optimizer minimizes
        return policy_loss - self.params['entropy']*entropy_loss

    def test(self, env, render=False):
        obs, done, ep_reward = env.reset(), False, 0
        i=0
        print('testing')
        # print(obs[None, :])
        while not done:
            action, _ = self.model.action_value(obs[None, :])
            # print('action', action)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            # print(i,': ',len([x for x in obs if x==9]))
            i+=1
            if i>=200:
                done=True
                print('Good: ',len([x for x in obs if x==3]))
            #     print('done')
            # elif i%10==0:
            #     print('-- ',i)
            if render:
                env.render()
                print("%d out of 200" % ep_reward)
                print(obs[None, :])
        return ep_reward
    
    
    def train(self, env, batch_sz=40, updates=1000):
        # storage helpers for a single batch of data
        actions = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        observations = np.empty((batch_sz,) + env.observation_space.shape)
        # training loop: collect samples, send to optimizer, repeat updates times
        ep_rews = [0.0]
        next_obs = env.reset()
        for update in range(updates):
            print('!=-100: ',len([x for x in rewards if x>0]),'/',len(rewards))
            if update%20==0:
                print(update)
            if update%50==0 and update>0:
                print("%d out of 200" % self.test(env))
                
            for step in range(batch_sz):
                observations[step] = next_obs.copy()
                actions[step], values[step] = self.model.action_value(next_obs[None, :])
                actions[step] = env.cardgood(actions[step])
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

model=Model(40)

env =envn(1)

agent = A2CAgent(model)
# rewards_sum = agent.test(env)
# print("%d out of 200" % rewards_sum) # 18 out of 200
rewards_history = agent.train(env, updates=1000)
print("Finished training, testing...")
print("%d out of 200" % agent.test(env)) # 200 out of 200