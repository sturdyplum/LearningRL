import numpy as np
import sys
import matplotlib.pyplot as plt
if "../" not in sys.path:
  sys.path.append("../")

def findLastMax(ar):
    return len(ar) - 1 - np.argmax(np.flip(ar))

def value_iteration_for_gamblers(p_h, theta=0.00001, discount_factor=1.0):
    """
    Args:
        p_h: Probability of the coin coming up heads
    """

    V = np.zeros(100)
    policy = np.zeros(100)

    while True:
        delta = 0
        for state in range(100):
            money_left = state
            action_values = np.zeros(99)
            for action in range(money_left + 1):
                if money_left + action > 100 or action > money_left:
                    continue

                action_values[action] += p_h * (1 if money_left + action == 100 else V[state+action])
                action_values[action] += (1.0 - p_h) * V[state-action]

            new_best = np.max(action_values)
            delta = max(delta, np.abs(new_best - V[state]))
            V[state] = new_best
        if delta < theta:
            break

        for state in range(100):
            money_left = state
            action_values = np.zeros(99)
            for action in range(99):
                if money_left + action > 100 or action > money_left:
                    continue

                action_values[action] += p_h * (1 if money_left + action == 100 else V[state+action])
                action_values[action] += (1.0 - p_h) * V[state-action]

            new_best = findLastMax(action_values)
            policy[state] = new_best


    return policy, V


policy, v = value_iteration_for_gamblers(0.25)

print("Optimized Policy:")
print(policy)
print("")

print("Optimized Value Function:")
print(v)
print("")


# Plotting Final Policy (action stake) vs State (Capital)

# x axis values
x = range(100)
# corresponding y axis values
y = v[:100]

# plotting the points
plt.plot(x, y)

# naming the x axis
plt.xlabel('Capital')
# naming the y axis
plt.ylabel('Value Estimates')

# giving a title to the graph
plt.title('Final Policy (action stake) vs State (Capital)')

# function to show the plot
plt.show()

# Plotting Capital vs Final Policy

# x axis values
x = range(100)
# corresponding y axis values
y = policy

# plotting the bars
plt.bar(x, y, align='center', alpha=0.5)

# naming the x axis
plt.xlabel('Capital')
# naming the y axis
plt.ylabel('Final policy (stake)')

# giving a title to the graph
plt.title('Capital vs Final Policy')

# function to show the plot
plt.show()
