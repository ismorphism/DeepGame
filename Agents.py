from builtins import *

import random
from abc import abstractmethod
from collections import defaultdict
from functools import partial

import numpy as np
import maci.utils as utils
from copy import deepcopy
from abc import ABCMeta, abstractmethod
import nashpy as nash



class Agent(object):
    __metaclass__ = ABCMeta

    def __init__(self, name, id_, action_num, env):
        self.name = name
        self.id_ = id_
        self.action_num = action_num
        # len(env.action_space[id_])
        # self.opp_action_space = env.action_space[0:id_] + env.action_space[id_:-1]

    def set_pi(self, pi):
        # assert len(pi) == self.action_num
        self.pi = pi

    def done(self, env):
        pass

    @abstractmethod
    def act(self, s, exploration, env):
        pass

    def update(self, s, a, o, r, s2, env):
        pass

    @staticmethod
    def format_time(n):
        return ''
        # s = humanfriendly.format_size(n)
        # return s.replace(' ', '').replace('bytes', '').replace('byte', '').rstrip('B')

    def full_name(self, env):
        return '{}_{}_{}'.format(env.name, self.name, self.id_)


class StationaryAgent(Agent):
    def __init__(self, id_, action_num, env, pi=None):
        super().__init__('stationary', id_, action_num, env)
        if pi is None:
            pi = np.random.dirichlet([1.0] * self.action_num)
        self.pi = np.array(pi, dtype=np.double)
        StationaryAgent.normalize(self.pi)

    def act(self, s, exploration, env):
        if self.verbose:
            print('pi of agent {}: {}'.format(self.id_, self.pi))
        return StationaryAgent.sample(self.pi)

    @staticmethod
    def normalize(pi):
        minprob = np.min(pi)
        if minprob < 0.0:
            pi -= minprob
        pi /= np.sum(pi)

    @staticmethod
    def sample(pi):
        return np.random.choice(pi.size, size=1, p=pi)[0]


class RandomAgent(StationaryAgent):
    def __init__(self, id_, action_num, env):
        assert action_num > 0
        super().__init__(id_, env, action_num, pi=[1.0 / action_num] * action_num)
        self.name = 'random'


class QAgent:
    def __init__(self, id_, action_num, n_states, env, alpha_decay_steps=10000., alpha=0.1, gamma=0.95, epsilon=0.5, verbose=True):
        self.action_num = action_num
        self.id_ = id_
        self.Q = np.zeros((n_states, self.action_num))
        # self.R = defaultdict(partial(np.zeros, self.action_num))
        self.count_R = defaultdict(partial(np.zeros, self.action_num))
        self.epsilon = epsilon
        self.alpha_decay_steps = alpha_decay_steps
        self.gamma = gamma
        self.alpha = alpha
        self.epoch = 0
        self.pi = defaultdict(partial(np.random.dirichlet, [1.0] * self.action_num))
        self.verbose = verbose
        self.pi_history = [deepcopy(self.pi)]

    def update(self, s, a, r, s2, env, done=False):
        V = self.val(s2)
        if done:
            self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * r
        else:
            self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * (r + self.gamma * V - self.Q[s, a])
        print(self.epoch)
        self.update_policy(s, a, env)
        self.epoch += 1

    def val(self, s):
        return np.max(self.Q[s])

    def update_policy(self, s, a, env):
        Q = self.Q[s]
        self.pi[s] = (Q == np.max(Q)).astype(np.double)

    def act(self, s, exploration):
        if exploration and random.random() < self.epsilon:
            return np.random.choice(np.arange(self.action_num), 1)[0]
        else:
            if self.verbose:
                print('Agent {}--------------'.format(self.id_))
                print('Q of agent {}: state {}: {}'.format(self.id_, s, str(self.Q[s])))
                print('pi of agent {}: state {}: {}'.format(self.id_, s, self.pi[s]))
                # print('payoff of agent {}: state {}: {}'.format(self.id_, s, self.R[s]))
                print('Agent {}--------------'.format(self.id_))
            return np.argmax(self.Q[s])


class NashAgent:
    def __init__(self, id_, action_num, n_states, env, alpha_decay_steps=10000., alpha=0.1, gamma=0.95, epsilon=0.5, verbose=True):
        self.action_num = action_num
        self.id_ = id_
        self.Q = np.zeros((2, n_states, self.action_num, self.action_num))
        self.Q_alone = np.zeros((n_states, self.action_num))
        self.epsilon = epsilon
        self.gamma = gamma
        self.base_alpha = alpha
        self.epoch = 0
        self.pi = defaultdict(partial(np.random.dirichlet, [1.0] * self.action_num))
        self.verbose = verbose
        self.history = np.zeros((n_states, self.action_num, self.action_num))
        # self.pi_history = [deepcopy(self.pi)]

    def update_nash(self, s, a, o, nash_actions, rewards, s2, env, done=False):
        self.history[s, a, o] += 1
        nash_actions_local = nash_actions[self.id_]
        V = [0, 0]
        if nash_actions_local is not None:
            V[0] = self.Q[0, s2, nash_actions_local[0], nash_actions_local[1]]
            V[1] = self.Q[1, s2, nash_actions_local[1], nash_actions_local[0]]
        else:
            V[0] = self.val(s2, agent_num=0)
            V[1] = self.val(s2, agent_num=1)

        alpha = self.base_alpha / (self.history[s, a, o])

        # alpha = self.base_alpha

        # V_alone = self.val(s2)
        # if done:
        #     self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * r
        # else:
        #     self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * (r + self.gamma * V - self.Q[s, a])
        if done:
            # self.Q_alone[s, a] = (1 - self.alpha) * self.Q_alone[s, a] + self.alpha * r
            self.Q[0, s, a, o] = (1 - alpha)*self.Q[0, s, a, o] + alpha*rewards[self.id_]  # my own Q
            self.Q[1, s, o, a] = (1 - alpha)*self.Q[1, s, o, a] + alpha*rewards[1-self.id_]  # Q of an other one
        else:
            self.Q[0, s, a, o] = (1 - alpha)*self.Q[0, s, a, o] + alpha*(rewards[self.id_] + self.gamma * V[0])
            self.Q[1, s, o, a] = (1 - alpha)*self.Q[1, s, o, a] + alpha*(rewards[1-self.id_] + self.gamma * V[1])
        print(self.epoch)
        self.epoch += 1

    def val(self, state, agent_num):
        return np.max(self.Q[agent_num, state])

    def val_alone(self, s):
        return np.max(self.Q[s])

    # def update_policy(self, s, a, game):
    #     self.pi[s] = utils.softmax(np.sum(np.multiply(self.Q[s], self.opponent_best_pi[s]), 1))
    #     self.pi_history.append(deepcopy(self.pi))
    #     self.opponent_best_pi_history.append(deepcopy(self.opponent_best_pi))
    #     print('opponent pi of {}: {}'.format(self.id_, self.opponent_best_pi))

    def nash_act(self, s, agent_id):
        A = self.Q[0, s]
        B = self.Q[1, s]
        rps = nash.Game(A, B)
        choices = [0, 0]
        eqs = list(rps.support_enumeration())
        nash_length = len(eqs)
        try:
            nash_choice = np.random.choice(np.arange(nash_length), 1)[0]
            eqs_0 = eqs[nash_choice][0].tolist()
            choices[0] = np.random.choice(np.arange(len(eqs_0)), p=eqs_0)
            eqs_1 = eqs[nash_choice][1].tolist()
            choices[1] = np.random.choice(np.arange(len(eqs_1)), p=eqs_1)
            return choices
        except ValueError:
            return None

    def act(self, s, exploration):
        if exploration and random.random() < self.epsilon:
            return np.random.choice(np.arange(self.action_num), 1)[0]
        else:
            choice = np.unravel_index(np.argmax(self.Q[0, s], axis=None), self.Q[0, s].shape)[0]
            if self.verbose:
                print('Agent {}--------------'.format(self.id_))
                print('Q of agent {}: state {}: {}'.format(self.id_, s, str(self.Q[0, s])))
                print('pi of agent {}: state {}: {}'.format(self.id_, s, self.pi[s]))
                # print('payoff of agent {}: state {}: {}'.format(self.id_, s, self.R[s]))
                print('Agent {}--------------'.format(self.id_))
            return choice
