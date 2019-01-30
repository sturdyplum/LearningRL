import numpy as np
import pprint
import sys
if "../" not in sys.path:
  sys.path.append("../")
from lib.envs.gridworld import GridworldEnv
pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()

def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """


    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])

    # Implement!

    while True:
        delta = 0
        for state in range(env.nS):
            action_values = np.zeros(env.nA)
            for action in range(env.nA):

                for prob, next_state, reward, _ in env.P[state][action]:
                    action_values[action] += prob * (reward + V[next_state] * discount_factor)

            best_action = np.max(action_values)
            delta = max(delta, np.abs(V[state] - best_action))
            V[state] = best_action
        if delta < theta:
            break

    for state in range(env.nS):
        action_values = np.zeros(env.nA)
        for action in range(env.nA):

            for prob, next_state, reward, _ in env.P[state][action]:
                action_values[action] += prob * (reward + V[next_state] * discount_factor)

        best_action = np.argmax(action_values)
        policy[state] = np.eye(env.nA)[best_action]


    return policy, V

policy, v = value_iteration(env)

print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")

# Test the value function
expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
