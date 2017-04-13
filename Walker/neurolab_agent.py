import neurolab as nl
import numpy as np


class NeuroLabLearningAgent(object):
    def __init__(self, observation_space_dim, action_space,
                 learning_rate=0.1,
                 discount=0.99,
                 exploration_rate=0.5,
                 exploration_decay_rate=0.99,
                 batch_size=10,
                 hidden1=4,
                 hidden2=3,
                 minInput=-10,
                 maxInput=10,
                 nnType="newff"):
                 #newff==multi-layer perceptron, newelm=elman recurrent network

        # Create train samples
        self.input_size=observation_space_dim
        self.output_size=4
        self.num_hidden1=hidden1
        self.num_hidden2=hidden2
        self._batch_size=batch_size
        self._xs=[]
        self._ys=[]
        self._i=0
        self.minInp=minInput
        self.maxInp=maxInput
        self.action_space=action_space

        #create multi-layer network with input_layer=inputs, hidden, and output = actions
        inpLayer = []
        # create list of ranges for input layer
        for f in range(self.input_size):
            inpLayer.append([self.minInp, self.maxInp])
        #can explicitly set transfer function
        #self.net = nl.net.newff(inpLayer, [self.num_hidden, self.output_size],
        #                        [nl.net.trans.TanSig(), nl.net.trans.SoftMax()])
        #default TanSig()
        self.net = nl.net.newff(inpLayer, [self.num_hidden1, # more layers here
                                            self.num_hidden2,
                                            self.output_size])
        #recurrent net
        if(nnType == "newelm"):
            self.net = nl.net.newelm(inpLayer, [self.num_hidden1,
												 # self.num_hidden2,
												 self.output_size])
        
        self.net.trainf = nl.train.train_gdx
        #gd, gdm, rprop: https://en.wikipedia.org/wiki/Rprop
        #self.net.errorf=nl.net.error.MSE
        self.net.init()
        self._n_actions = 4
        self._learning_rate = learning_rate
        self._discount = discount
        self._exploration_rate = exploration_rate
        self._exploration_decay = exploration_decay_rate
        self._updates_count = 0

    def save(self, filename):
        self.net.save(filename)

    def load(self, filename):
        if self.net is None:
            self.net = nl.load(filename)

    def reset(self):
        self._exploration_rate *= self._exploration_decay

    def act(self, observation):
        if np.random.random_sample() < self._exploration_rate:
            return self.action_space.sample()
        else:
            # Simulate network
            scores = self.net.sim([observation])
            # out = scores.argmax()
            # argmax=None
            # max=-float("inf")
            # for i in range(scores.shape[0]):
            #     temp = np.sum(scores[i])
            #     if temp>max:
            #         max=temp
            #         argmax=i
            # return scores[argmax]
            action=scores[0]
            action=action.tolist()
            # action[action<=0 ]=0
            action[action < -0.5] = -1
            action[action < 0.5 and action>=-0.5] = 0
            action[action >0] = 1
            return action

    def update(self, observation, action, new_observation, reward, forceRetrain=False):
        # does not update immediately, collects data and updates in batches
        # current predicted rewards from --new--- observation
        scores_future = self.net.sim([new_observation])[0]
        discounted = []
		
        for s in scores_future:
            ds = reward + self._discount*s
            discounted.append(ds)

        self._xs.append(observation)
        self._ys.append(discounted)

        self._i+=1
        if(self._i >= self._batch_size or forceRetrain):
            self._i=0

            tempx=[]
            tempy=[]
            tempx.extend(self._xs)
            tempy.extend(self._ys)

            self.net.train(tempx,tempy,
							epochs=200,
							show=0,
							lr=self._learning_rate,
							# lr_dec=0.99,
							# mc=0.9
                           )
            self._xs = []
            self._ys = []

    def discretize(self,observation):
        observation = observation.tolist()
        # temp=observation[0:3]
        observation[observation <= -1] = -1
        # observation[observation <= 0] = 0
        observation[observation < 1 and observation > -1] = 0
        observation[observation >=1] = 1
        # observation[0:3]=temp
        return np.asarray(observation)
