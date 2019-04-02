from maci.environments import grid_world
from Agents import QAgent, NashAgent

import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import random
import multiprocessing

seed = 0
np.random.seed(0)
random.seed(0)


def calc(i):
    return 2 * i



def nash_acting(agent, state_next):
    return agent.nash_act(state_next, agent.id_)



def nash_updating(agent, state_transformed, actions, nash_actions, rewards, state_transformed_n, done):
    agent.update_nash(state_transformed, actions[agent.id_], actions[1-agent.id_], nash_actions, rewards, state_transformed_n, done)
    agent.epsilon *= 0.999991 #for 5x5
    # agent.epsilon *= 0.9999
    return agent



def run():

    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-gs', '--grid_size', default=3, type=int)
    parser.add_argument('-ga', '--gamma', default=0.99, type=float)
    parser.add_argument('-lr', '--learning_rate', default=0.95, type=float)
    parser.add_argument('-e', '--epsilon', default=1.0, type=float)
    parser.add_argument('-iters', '--iterations', default=1000, type=int)
    parser.add_argument('-m', '--method', default='Q', type=str)
    parser.add_argument('-exp', '--exploration_type', default='e-greedy', type=str)
    args = parser.parse_args()

    start_time = time.perf_counter()
    agent_num = 2
    action_num = 4
    grid_world_size = args.grid_size
    agents = []

    n_states = grid_world_size**(2*agent_num)

    state_matrix = np.zeros((grid_world_size, grid_world_size, grid_world_size, grid_world_size), dtype=int)
    cnt = 0
    for i in range(grid_world_size):
        for j in range(grid_world_size):
            for k in range(grid_world_size):
                for l in range(grid_world_size):
                    state_matrix[i, j, k, l] = int(cnt)
                    cnt += 1

    # state_matrix = np.zeros((grid_world_size, grid_world_size))
    # for i in range(grid_world_size):
    #     for j in range(grid_world_size):
    #         state_matrix[i, j] = cnt
    #         cnt += 1


    def draw_grid(env):
        Grid = np.zeros((grid_world_size, grid_world_size))
        Grid[env.agents[0][0], env.agents[0][1]] = 3
        Grid[env.agents[1][0], env.agents[1][1]] = 7
        Grid[env.goals[0][0], env.goals[0][1]] = 10
        Grid[env.goals[1][0], env.goals[1][1]] = 10
        Grid[env.obstacle[0], env.obstacle[1]] = 20
        plt.imshow(Grid)
        plt.set_cmap('hot_r')
        plt.title("Agent #0 is Yellow, Agent #1 is Red")
        plt.show()
        # input()

    iteration_num = args.iterations

    env = grid_world.GridGame(gridsize=grid_world_size, n_agents=agent_num, seed=seed)
    env.reset()
    # rewards_hist = np.zeros((iteration_num, agent_num))

    draw_grid(env)


    if args.method == 'Q':
        for i in range(agent_num):
            agent = QAgent(i, action_num, n_states, env, alpha=args.learning_rate, gamma=args.gamma,
                           epsilon=args.epsilon,
                           exp_type=args.exploration_type)
            agents.append(agent)
    elif args.method == 'NashQ':
        for i in range(agent_num):
            agent = NashAgent(i, action_num, n_states, env, alpha=args.learning_rate, gamma=args.gamma,
                              epsilon=args.epsilon,
                              dyna=True,
                              planning_steps=1, exp_type=args.exploration_type)
            agents.append(agent)

    states = env.getSensors()
    states_transformed = [state_matrix[num[0], num[1], num[2], num[3]] for num in states]

    successes_0 = 0
    successes_1 = 0
    success_story_0 = []
    success_story_1 = []
    exploration = True
    story = 0
    action_history = {}
    action_history[story] = []
    success_lengths = []
    cnt = 0

    num_cpus = multiprocessing.cpu_count() - 1
    pool = multiprocessing.Pool(processes=num_cpus)

    for i in range(0, iteration_num):
        cnt += 1
        actions = np.array([agent.act(state_transformed, exploration) for state_transformed, agent in zip(states_transformed, agents)])
        states_n, rewards, dones = env.step(actions)
        states_transformed_n = [state_matrix[num[0], num[1], num[2], num[3]] for num in states_n]
        if args.method == 'NashQ':
            nash_actions = [pool.apply_async(nash_acting, args=(agent, state_next)) for state_next, agent in zip(states_transformed_n, agents)]
            nash_actions = [p.get() for p in nash_actions]

            agents = [pool.apply_async(nash_updating, args=(agent, state_transformed, actions, nash_actions, rewards, state_transformed_n, done)) for agent, state_transformed, state_transformed_n, done in zip(agents, states_transformed, states_transformed_n, dones)]
            agents = [p.get() for p in agents]

            # for j, (state_transformed, reward, state_transformed_n, agent, done) in enumerate(zip(states_transformed, rewards, states_transformed_n, agents, dones)):
            #     agent.update_nash(state_transformed, actions[j], actions[1-j], nash_actions, rewards, state_transformed_n, env, done=done)  # Nash Q-learning agent
            #     agent.epsilon *= 0.999991 #TODO: change the val to [0.99991, 0.99999) Previous was 0.9999 (very fast drop), 0.99999 (very slow)
        elif args.method == 'Q':
            for j, (state_transformed, reward, state_transformed_n, agent, done) in enumerate(
                    zip(states_transformed, rewards, states_transformed_n, agents, dones)):
                agent.update(state_transformed, actions[j], reward, state_transformed_n, env, done=done) # vanilla Q-learning agent
                agent.epsilon *= 0.9999 #TODO: change the val to [0.99991, 0.99999)
        action_history[story].append(actions)
        states_transformed = states_transformed_n

        # plt.figure()
        #
        # print(env.agents)
        # draw_grid(env)

        if dones[0] or dones[1]:
            story += 1
            action_history[story] = []
            successes_0 += int(dones[0])
            successes_1 += int(dones[1])
            success_lengths.append(cnt)
            cnt = 0
            env.reset()

        print('Iter #{}'.format(i))
        print('Rewards are ', rewards, 'Success ', dones)
        print('Exploration is ', agents[0].epsilon)

        if i % 100 == 0:
            # plt.figure()
            success_story_0.append(successes_0)
            successes_0 = 0
            # plt.subplot(1, 2, 1)
            # plt.plot(rewards_hist[:, 0])
            # plt.plot(success_story_0)
            # plt.title("First agent success history")
            # plt.grid()
            # plt.xlabel('Number of 100 games')
            # plt.subplot(1, 2, 2)
            success_story_1.append(successes_1)
            successes_1 = 0
            # plt.plot(success_story_1)
            # plt.plot(rewards_hist[:, 1])
            # plt.title("Second agent success history")
            # plt.grid()
            # plt.xlabel('Number of 100 games')
            # plt.show()

    print('----------------------')
    print('\n')
    print('RESULTS')
    print('Total working time is ', time.perf_counter() - start_time, ' seconds')
    print('Average success time for 100 games is ', np.mean(success_lengths[-100:]), ' steps')
    print('The fastest route is ', min(success_lengths[-100:]), ' steps ')
    print('Duration of last 10 games: ', success_lengths[-10:], ' steps ')
    print('First agent is  ', success_story_0[-100:], ' successes ')
    print('Second agent is  ', success_story_1[-100:], ' successes ')
    input('Let\'s draw \n')
    title = 'Average amount of steps (for both agents) = ' + '{:.3f}'.format(np.mean(success_lengths)) + ' steps' + '\n' + \
            'Number of steps is {}'.format(iteration_num)
    plt.suptitle(title, fontsize="x-large")
    plt.subplot(1, 3, 1)
    plt.plot(success_story_0)
    plt.title("First agent success history")
    plt.grid()
    plt.xlabel('Number of 100 games')
    plt.subplot(1, 3, 2)
    plt.plot(success_story_1)
    plt.title("Second agent success history")
    plt.grid()
    plt.xlabel('Number of 100 games')
    plt.subplot(1, 3, 3)
    plt.plot(success_lengths[-100:])
    plt.title("Lengths of successes (steps)")
    plt.grid()
    plt.xlabel('Amount of successes')
    plt.show()
    print(action_history.keys())


if __name__ == '__main__':
    run()
