import numpy as np
import random

no_shops = 4
no_items = 2
reward_buying = 50

prices = np.random.randint(20, 100, size=(no_shops, no_items))
availability = np.random.rand(no_shops, no_items) > 0.3
distances = np.random.randint(1, 10, size=(no_shops, no_shops))
np.fill_diagonal(distances, 0)

def get_reward(shop, items_bought, new_items):
    new_count = sum(new_items) - sum(items_bought)
    price_penalty = sum(
        prices[shop][i]
        for i in range(no_items)
        if new_items[i] and not items_bought[i]
    )
    return new_count * reward_buying - price_penalty

def take_action(state, action):
    shop, items_bought = state
    new_items = list(items_bought)
    for i in range(no_items):
        if not items_bought[i] and availability[action][i]:
            new_items[i] = 1
    reward = get_reward(action, items_bought, tuple(new_items))
    reward -= distances[shop][action]
    new_state = (action, tuple(new_items))
    return new_state, reward

def epsilon_greedy(Q, state, epsilon=0.1):
    if random.random() < epsilon:
        return random.randint(0, no_shops - 1)
    return np.argmax(Q[state])

def q_learning(episodes=500, steps=100, alpha=0.1, gamma=0.9, epsilon=0.2):
    state_space = []
    for s in range(no_shops):
        for i1 in [0, 1]:
            for i2 in [0, 1]:
                state_space.append((s, (i1, i2)))
    
    Q = {s: np.zeros(no_shops) for s in state_space}
    all_rewards = []

    for ep in range(episodes):
        state = random.choice(state_space)
        total_reward = 0

        for _ in range(steps):
            if state[1] == (1, 1):
                break

            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward = take_action(state, action)

            if next_state not in Q:
                Q[next_state] = np.zeros(no_shops)
            
            Q[state][action] += alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state][action]
            )
            total_reward += reward
            state = next_state

        all_rewards.append(total_reward)

    return Q, all_rewards

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Prices:\n", prices)
    print("Availability:\n", availability)
    print("Distances:\n", distances)

    Q, rewards = q_learning()

    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-learning Progress")
    plt.show()

    print("\nBest Policy:")
    for state in Q:
        best_action = int(np.argmax(Q[state]))
        print(f"State {state} → Best Action: Go to Shop {best_action}")
