import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim


class dqn(object):

    def __init__(self, ndim_input, num_actions, discount=0.99, model_path=None):
        self.num_actions = num_actions

        self.discount = discount
        self.sess = tf.Session()

        self.inputs_tf = tf.placeholder(tf.float32, [None, ndim_input])
        self.q_values_tf = self.get_q_net(self.inputs_tf, num_actions)

        self.action_mask_tf = tf.placeholder(tf.float32, [None, num_actions])
        self.target_tf = tf.placeholder(tf.float32, [None, 1])

        loss = tf.reduce_mean(
            tf.pow(self.q_values_tf - self.target_tf, 2) * self.action_mask_tf)

        optim = tf.train.RMSPropOptimizer(1e-3)

        self.train_op = slim.learning.create_train_op(loss, optim)

        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()

        if model_path is not None:
            self.saver.restore(self.sess, model_path)

    def get_q_net(self, inputs, num_actions):
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.layer_norm):
            net = slim.fully_connected(self.inputs_tf, 128, scope="l1")
            net = slim.fully_connected(net, 128, scope="l2")
            return slim.fully_connected(
                net, num_actions, normalizer_fn=None, activation_fn=None, scope="l3")

    def act(self, obs):
        q_values_np = self.sess.run(self.q_values_tf, {self.inputs_tf: obs})
        return np.argmax(q_values_np, 1)

    def get_targets(self, reward, newobs, mask):
        q_values_np = self.sess.run(
            self.q_values_tf, {self.inputs_tf: newobs})

        return np.max(q_values_np, 1, keepdims=True) * mask * self.discount + reward

    def update(self, obs, action, reward, newobs, mask):
        q_targets = self.get_targets(reward, newobs, mask)

        onehot_actions = np.zeros((action.shape[0], self.num_actions))
        onehot_actions[np.arange(action.shape[0]),
                       action.astype('int')[:, 0]] = 1

        self.sess.run(self.train_op, {self.inputs_tf: obs,
                                      self.target_tf: q_targets,
                                      self.action_mask_tf: onehot_actions})

    def save(self, save_path="/tmp/model.ckpt"):
        self.saver.save(self.sess, save_path)
