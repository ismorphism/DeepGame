from builtins import *

import random
from abc import abstractmethod
from collections import defaultdict
from functools import partial

import numpy as np
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
    def __init__(self, id_, action_num, n_states, env, alpha_decay_steps=10000., alpha=0.1, gamma=0.95, epsilon=0.5, verbose=True, exp_type='e-greedy'):
        self.action_num = action_num
        self.id_ = id_
        self.Q = np.zeros((n_states, self.action_num))
        self.startTau = self.tau = 1.0
        self.endTau = 0.1
        self.anneling_steps = 500
        self.pre_trained = 100
        self.exploration = exp_type
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
        # print(self.epoch)
        self.epoch += 1

    def val(self, s):
        return np.max(self.Q[s])

    def act(self, s, exploration):

        if self.exploration == 'e-greedy':
            if random.random() < self.epsilon:
                return np.random.choice(np.arange(self.action_num), 1)[0]
            else:
                if self.verbose:
                    print('Agent {}--------------'.format(self.id_))
                    print('Q of agent {}: state {}: {}'.format(self.id_, s, str(self.Q[s])))
                    # print('payoff of agent {}: state {}: {}'.format(self.id_, s, self.R[s]))
                    print('Agent {}--------------'.format(self.id_))
                return np.argmax(self.Q[s])

        elif self.exploration == 'boltzmann':
            stepDrop = (self.startTau - self.endTau) / self.anneling_steps
            if self.tau > self.endTau and self.epoch > self.pre_trained:
                self.tau -= stepDrop
            probs = np.exp(self.Q[s]/self.tau)/np.sum(np.exp(self.Q[s]/self.tau))
            return np.random.choice(np.arange(self.action_num), p=probs)


class NashAgent:
    def __init__(self, id_, action_num, n_states, env, alpha_decay_steps=10000., alpha=0.1, gamma=0.95, epsilon=0.5, verbose=True, dyna=False, planning_steps=1, exp_type='e-greedy'):
        self.action_num = action_num
        self.id_ = id_
        self.Q = np.zeros((2, n_states, self.action_num, self.action_num))
        self.Q_alone = np.zeros((n_states, self.action_num))
        self.epsilon = epsilon
        self.startTau = self.tau = 1.0
        self.endTau = 0.1
        self.anneling_steps = 500
        self.pre_trained = 1000
        self.exploration = exp_type
        self.dyna_update = False #let's add N planning updates. Let's call it Dyna-N
        if dyna:
            self.dyna_update = True
            self.planning_steps = planning_steps
            self.dyna_model = {}
            self.rewards_dyna = {}
        self.gamma = gamma
        self.base_alpha = alpha
        self.epoch = 0
        self.pi = defaultdict(partial(np.random.dirichlet, [1.0] * self.action_num))
        self.verbose = verbose
        self.history_alpha = np.zeros((n_states, self.action_num, self.action_num))
        self.history_act = np.zeros((n_states, 1))
        # self.pi_history = [deepcopy(self.pi)]

    def update_nash(self, s, a, o, nash_actions, rewards, s2, done=False):

        if self.dyna_update:
            if s2 not in self.dyna_model:
                self.dyna_model[s2] = [[s, a, o]]
                self.rewards_dyna[s2] = [[rewards, done]]
            else:
                if [s, a, o] not in self.dyna_model[s2]:
                    self.dyna_model[s2].append([s, a, o])
                    self.rewards_dyna[s2].append([rewards, done])

                # except ValueError:
                #     print([s, a, o, rewards, done])
                #     print('FCKK')
                #     print(self.dyna_model[s2])
                #     input()

        self.history_alpha[s, a, o] += 1
        nash_actions_local = nash_actions[self.id_]
        V = [0, 0]
        if None not in nash_actions_local:
            V[0] = self.Q[0, s2, nash_actions_local[0], nash_actions_local[1]]
            V[1] = self.Q[1, s2, nash_actions_local[1], nash_actions_local[0]]
            # for i in range(len(nash_actions_local)):
            #     for j in range(len(nash_actions_local)):# it's probabilities now
            #         V[0] += self.Q[0, s2, i, j]*nash_actions_local[0][i]*nash_actions_local[1][j]
            # for i in range(len(nash_actions_local)):
            #     for j in range(len(nash_actions_local)):# it's probabilities now
            #         V[1] += self.Q[1, s2, j, i]*nash_actions_local[0][i]*nash_actions_local[1][j]
        else:
            V[0] = self.val(s2, agent_num=0)
            V[1] = self.val(s2, agent_num=1)

        alpha = self.base_alpha / (self.history_alpha[s, a, o])

        if done:
            # self.Q_alone[s, a] = (1 - self.alpha) * self.Q_alone[s, a] + self.alpha * r
            self.Q[0, s, a, o] = (1 - alpha)*self.Q[0, s, a, o] + alpha*rewards[self.id_]  # my own Q
            self.Q[1, s, o, a] = (1 - alpha)*self.Q[1, s, o, a] + alpha*rewards[1-self.id_]  # Q of an other one
        else:
            # try:
            self.Q[0, s, a, o] = (1 - alpha)*self.Q[0, s, a, o] + alpha*(rewards[self.id_] + self.gamma * V[0])
            self.Q[1, s, o, a] = (1 - alpha)*self.Q[1, s, o, a] + alpha*(rewards[1-self.id_] + self.gamma * V[1])
            # except ValueError:
            #     print(V)
            #     print(nash_actions_local)
            #     print(nash_actions)

        # Dyna update d as addition symbol means Dyna
        if self.dyna_update:
            count_outer = 0
            count_inner = 0
            for state_key in self.dyna_model.keys():
                if count_outer <= self.planning_steps:
                    dyna_nash_actions = self.nash_act(state_key, self.id_)
                    V_dyna = [0, 0]
                    if None not in dyna_nash_actions:
                        V_dyna[0] = self.Q[0, state_key, dyna_nash_actions[0], dyna_nash_actions[1]]
                        V_dyna[1] = self.Q[1, state_key, dyna_nash_actions[1], dyna_nash_actions[0]]
                    else:
                        V_dyna[0] = self.val(state_key, agent_num=0)
                        V_dyna[1] = self.val(state_key, agent_num=1)

                    for num in range(len(self.dyna_model[state_key])):
                        if count_inner <= self.planning_steps:
                            d_s, d_a, d_o = self.dyna_model[state_key][num]
                            dywards, d_done = self.rewards_dyna[state_key][num]
                            if done:
                                self.Q[0, d_s, d_a, d_o] = (1 - alpha)*self.Q[0, d_s, d_a, d_o] + alpha*dywards[self.id_]  # my own Q
                                self.Q[1, d_s, d_o, d_a] = (1 - alpha)*self.Q[1, d_s, d_o, d_a] + alpha*dywards[1-self.id_]  # Q of an other one
                            else:
                                # TODO use matrix summation for speed-up the computations
                                self.Q[0, d_s, d_a, d_o] = (1 - alpha)*self.Q[0, d_s, d_a, d_o] + alpha*(dywards[self.id_] + self.gamma * V_dyna[0])
                                self.Q[1, d_s, d_o, d_a] = (1 - alpha)*self.Q[1, d_s, d_o, d_a] + alpha*(dywards[1-self.id_] + self.gamma * V_dyna[1])
                        else: break
                        count_inner += 1
                    count_outer += 1
                else: break

        # print(self.epoch)
        self.epoch += 1

    def val(self, state, agent_num):
        return np.max(self.Q[agent_num, state])

    def val_alone(self, s):
        return np.max(self.Q[s])

    def nash_act(self, s, agent_id):
        A = self.Q[0, s]
        B = self.Q[1, s]
        rps = nash.Game(A, B)
        choices = [0, 0]
        eqs = list(rps.support_enumeration())
        nash_length = len(eqs)
        try:
            # nash_choice = np.random.choice(np.arange(nash_length), 1)[0]  #Random choice of Nash-equilibrium
            nash_choice = 0
            eqs_0 = eqs[nash_choice][0].tolist()
            choices[0] = np.random.choice(np.arange(len(eqs_0)), p=eqs_0)
            eqs_1 = eqs[nash_choice][1].tolist()
            choices[1] = np.random.choice(np.arange(len(eqs_1)), p=eqs_1)
            # return [eqs_0, eqs_1]
            return choices
        except IndexError:
            return [None, None]

    def act(self, s, exploration):

        # eps = 1/(1 + self.history_act[s])
        # self.history_act[s] += 1

        if self.exploration == 'e-greedy':
            if random.random() < self.epsilon:
                return np.random.choice(np.arange(self.action_num), 1)[0]
            else:
                choice = np.unravel_index(np.argmax(self.Q[0, s], axis=None), self.Q[0, s].shape)[0]
                # A = self.Q[0, s]
                # B = self.Q[1, s]
                # rps = nash.Game(A, B)
                # eqs = list(rps.support_enumeration())
                # eqs_0 = eqs[0][0].tolist()
                # choice = np.random.choice(np.arange(len(eqs_0)), p=eqs_0)
                return choice
        elif self.exploration == 'boltzmann':
            stepDrop = (self.startTau - self.endTau) / self.anneling_steps
            if self.tau > self.endTau and self.epoch > self.pre_trained:
                self.tau -= stepDrop
            Q_max = np.max(self.Q[0, s], axis=1)/100
            probs = np.exp(Q_max/self.tau)/np.sum(np.exp(Q_max/self.tau))
            return np.random.choice(range(self.action_num), p=probs)
