import numpy as np
import pprint
import sys
if "../" not in sys.path:
  sys.path.append("../")
from lib.envs.gridworld import GridworldEnv
from PolicyEvalutaion import policy_eval

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()

def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:
        converged = True
        values = policy_eval_fn(policy, env)

        for state in range(env.nS):
            current_best = np.argmax(policy[state])

            new_values = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, _ in env.P[state][action]:
                    new_values[action] += prob * (discount_factor * values[next_state] + reward)

            new_best = np.argmax(new_values)

            if new_best != current_best:
                converged = False

            # Nice trick!
            policy[state] = np.eye(env.nA)[new_best]
        if converged:
            return policy, values



    return policy, np.zeros(env.nS)

policy, v = policy_improvement(env)
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
