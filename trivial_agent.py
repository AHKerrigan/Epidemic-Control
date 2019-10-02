import epi_env
import numpy as np

EPISODES = 11500

env = epi_env.Epi_Env()
running_reward = 0
f = open("Trivial-Karate-Time.cvs")
for episode in range(EPISODES):
    env.reset()
    for node in range(len(env.graph)):
        env.sim.change_inf_rate(node, env.beta_low)
        env.sim.change_rec_rate(node, env.delta_hi)
    
    done = False
    trivial_action = np.ones(env.action_space)

    sum_reward = 0
    while done != True:
        state, reward, done, _ = env.step(trivial_action)
        sum_reward += reward
    running_reward = (running_reward * 0.995) + (sum_reward * 0.005)

    print(running_reward)


