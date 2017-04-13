### Project 4 by Yunchuan Kong

import numpy as np
import tensorflow as tf

class neuralAgent(object):
    def __init__(self, n_obs=80*80,
                 h=200,
	             n_actions=3,										   
                 learning_rate=0.001,
                 gamma=0.99,
                 decay=0.99,
				 save_path='training.ckpt'):
				
        self.n_obs=n_obs
        self.h=h
        self.n_actions=n_actions
        self.learning_rate=learning_rate
        self.gamma=gamma
        self.decay=decay
        self.save_path=save_path
        self.episode_number=1
		
        def tf_discount_rewards(tf_r):
            discount_f = lambda a, v: a*self.gamma + v
            tf_r_reverse = tf.scan(discount_f, tf.reverse(tf_r,[True, False]))
            tf_discounted_r = tf.reverse(tf_r_reverse,[True, False])
            return tf_discounted_r

        def tf_policy_forward(x):
            self.h = tf.matmul(x, self.tf_model['W1'])
            self.h = tf.nn.relu(self.h)
            logp = tf.matmul(self.h, self.tf_model['W2'])
            p = tf.nn.softmax(logp)
            return p

        self.tf_model={}
        with tf.variable_scope('layer_one',reuse=False):
            xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(self.n_obs), dtype=tf.float32)
            self.tf_model['W1'] = tf.get_variable("W1", [self.n_obs, self.h], initializer=xavier_l1)
        with tf.variable_scope('layer_two',reuse=False):
            xavier_l2 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(self.h), dtype=tf.float32)
            self.tf_model['W2'] = tf.get_variable("W2", [self.h,self.n_actions], initializer=xavier_l2)
		
        self.tf_x = tf.placeholder(dtype=tf.float32, shape=[None, self.n_obs],name="tf_x")
        self.tf_y = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions],name="tf_y")
        self.tf_epr = tf.placeholder(dtype=tf.float32, shape=[None,1], name="tf_epr")
		
        self.tf_discounted_epr = tf_discount_rewards(self.tf_epr)
        self.tf_mean, self.tf_variance= tf.nn.moments(self.tf_discounted_epr, [0], shift=None, name="reward_moments")
        self.tf_discounted_epr -= self.tf_mean
        self.tf_discounted_epr /= tf.sqrt(self.tf_variance + 1e-6)

        self.tf_aprob = tf_policy_forward(self.tf_x)
        self.loss = tf.nn.l2_loss(self.tf_y - self.tf_aprob)
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=self.decay)
        tf_grads = optimizer.compute_gradients(self.loss, var_list=tf.trainable_variables(), grad_loss=self.tf_discounted_epr)
        self.train_op = optimizer.apply_gradients(tf_grads)
		
        self.sess = tf.InteractiveSession()
        tf.initialize_all_variables().run()
		
        self.saver = tf.train.Saver(tf.all_variables())
        self.xs=[]
        self.ys=[]
        self.rs=[]
        self.feed = None

    def act(self, features):
        self.feed = {self.tf_x: np.reshape(features, (1, -1))}
        aprob = self.sess.run(self.tf_aprob, self.feed)
        aprob = aprob[0, :]
        action = np.random.choice(self.n_actions, p=aprob)
        label = np.zeros_like(aprob)
        label[action] = 1

        self.xs.append(features)
        self.ys.append(label)

        return action, label

    def record(self,reward):
        self.rs.append(reward)

    def save(self):
        self.saver.save(self.sess, self.save_path, global_step=self.episode_number)
        print "model saved.".format(self.episode_number-1)

    def load(self):
        save_dir = '/'.join(self.save_path.split('/')[:-1])
        ckpt = tf.train.get_checkpoint_state(save_dir)
        load_path = ckpt.model_checkpoint_path
        self.saver.restore(self.sess, load_path)
        self.saver = tf.train.Saver(tf.all_variables())
        print "model loaded.".format(self.save_path)

    def update(self):
        self.episode_number+=1
        self.feed = {self.tf_x: np.vstack(self.xs), self.tf_epr: np.vstack(self.rs), self.tf_y: np.vstack(self.ys)}
        _ = self.sess.run(self.train_op,self.feed)
        self.xs=[]
        self.ys=[]
        self.rs=[]





