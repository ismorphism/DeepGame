import numpy as np
import copy
import random


class GridGame():

    def __init__(self, gridsize=3, n_agents=2, seed=0):

        self.availableActions = [0, 1, 2, 3]  # Corresponding to forward north, westa, south, east respectively.
        self.sizeofGridWorld = gridsize
        self.numberofGoals = 2
        self.numberofAgents = n_agents
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)


    def getSensors(self):
        # State: locations of all agents
        state = [np.r_[self.agents[0], self.agents[1]],
                 np.r_[self.agents[0], self.agents[1]]]
        return state

    def performAction(self, actions):
        tempPos = []
        self.prevAgents = copy.deepcopy(self.agents)
        self.isCollide = False
        for i in range(self.numberofAgents):
            tempPos.append(self.__move__(copy.deepcopy(self.agents[i]), actions[i]))
        if not self.__isCollideWithEachOther(tempPos):
            self.agents = tempPos

    def step(self, actions):
        #observation, reward, done, info
        self.performAction(actions)
        rewards = self.getJointReward()
        done = self.__isReachGoal()
        state_next = self.getSensors()
        return state_next, rewards, done


    def __move__(self, position, forward):
        if forward == 0:  # Move North
            position[1] += 1
        elif forward == 1:  # Move west
            position[0] -= 1
        elif forward == 2:  # Move south
            position[1] -= 1
        elif forward == 3:  # Move east
            position[0] += 1
        else:
            assert False, "Unexpected action"

        if position[0] >= self.sizeofGridWorld:
            position[0] = self.sizeofGridWorld - 1
        if position[0] < 0:
            position[0] = 0
        if position[1] >= self.sizeofGridWorld:
            position[1] = self.sizeofGridWorld - 1
        if position[1] < 0:
            position[1] = 0
        return position

    def __isCollideWithEachOther(self, tempPos):
        if (tempPos[0][0] == tempPos[1][0]) and (tempPos[0][1] == tempPos[1][1]):
            if (tempPos[0][0] != self.goals[0][0]) or (tempPos[0][1] != self.goals[0][1]):
                self.isCollide = True
                return True
            else:
                return False
        else:
            return False

    def __isReachGoal(self):
        # return boolean list, that determine if each agent reach each goal.
        irGoal = [False, False]
        if (self.agents[0][0] == self.goals[0][0]) and (self.agents[0][1] == self.goals[0][1]):  # For the first agent.
            irGoal[0] = True
            self.isReachGoal = True
        if (self.agents[1][0] == self.goals[1][0]) and (
                self.agents[1][1] == self.goals[1][1]):  # For the second agent.
            irGoal[1] = True
            self.isReachGoal = True
        return irGoal

    def reset(self):
        # self.agents = [np.array([0, 0]),
        #                np.array([2, 0])]
        # self.prevAgents = [np.array([0, 0]),
        #                    np.array([2, 0])]

        row_1 = np.random.choice([0, 1], 1)[0]
        column_1 = np.random.choice([0, 1], 1)[0]

        second_list_row = [1, 2]
        if row_1 in second_list_row:
            second_list_row.remove(row_1)

        second_list_column = [0, 1]
        if column_1 in second_list_row:
            second_list_column.remove(column_1)

        row_2 = np.random.choice(second_list_row, 1)[0]
        column_2 = np.random.choice(second_list_column, 1)[0]

        self.agents = [np.array([row_1, column_1]),
                       np.array([row_2, column_2])]

        self.prevAgents = [np.array([row_1, column_1]),
                       np.array([row_2, column_2])]

        self.isReachGoal = False
        self.goals = [np.array([2, 2]),
                          np.array([0, 2])]

    def getJointReward(self):
        jointRew = [0, 0]
        irGoal = self.__isReachGoal()
        if irGoal[0]:
            jointRew[0] = 100
        if irGoal[1]:
            jointRew[1] = 100
        if self.isCollide:
            jointRew[0] -= 1
            jointRew[1] -= 1

        return np.array(jointRew)


class PredatorPrey():
    pass


class Soccer():
    pass
