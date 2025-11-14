import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from rl_project import (
    no_shops,
    no_items,
    reward_buying,
    prices,
    availability,
    distances,
    take_action,
    q_learning,
)

st.set_page_config(
    page_title="Q-Grocery Shopping Solution",
    layout="wide",
)

st.title("Q-Grocery Shopping Solution")

def plot_shop_network(distances):
    G = nx.Graph()
    num_shops = len(distances)
    for i in range(num_shops):
        G.add_node(i, label=f"{i+1}")
    for i in range(num_shops):
        for j in range(i + 1, num_shops):
            if distances[i][j] > 0:
                G.add_edge(i, j, weight=distances[i][j])

    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(12, 8))
    node_labels = {i: f"{i+1}" for i in range(num_shops)}
    nx.draw(G, pos, labels=node_labels, with_labels=True, node_color="skyblue",
            node_size=700, ax=ax, font_size=12, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    ax.set_title("Shop Network (Distances)")
    st.pyplot(fig)
    plt.close(fig)

def display_q_table(Q):
    if isinstance(Q, dict):
        q_df = pd.DataFrame.from_dict(Q, orient='index')
    else:
        q_df = pd.DataFrame(Q)
    q_df = q_df.round(2)
    q_df.columns = [f"Shop {i+1}" for i in range(q_df.shape[1])]
    if isinstance(Q, dict):
        q_df.index = [f"{state}" for state in Q.keys()]
    else:
        q_df.index = [f"State {i+1}" for i in range(q_df.shape[0])]
    st.dataframe(q_df.style.background_gradient(cmap="viridis").format("{:.2f}"), use_container_width=True)

env_tab, train_tab, sim_tab, graph_tab = st.tabs(["Environment", "Train Agent", "Simulation", "Shop Network"])

with env_tab:
    st.subheader("Shop Environment Overview")

    col1, col2 = st.columns(2)

    with col1:
        prices_df = pd.DataFrame(prices)
        prices_df.columns = [f"Item {i+1}" for i in range(prices_df.shape[1])]
        prices_df.index = [f"Shop {i+1}" for i in range(prices_df.shape[0])]
        st.dataframe(prices_df, use_container_width=True)

    with col2:
        avail_df = pd.DataFrame(availability)
        avail_df.columns = [f"Item {i+1}" for i in range(avail_df.shape[1])]
        avail_df.index = [f"Shop {i+1}" for i in range(avail_df.shape[0])]
        st.dataframe(avail_df, use_container_width=True)

    dist_df = pd.DataFrame(distances)
    dist_df.columns = [f"Shop {i+1}" for i in range(dist_df.shape[1])]
    dist_df.index = [f"Shop {i+1}" for i in range(dist_df.shape[0])]
    st.dataframe(dist_df, use_container_width=True)

with train_tab:
    st.subheader("Train Q-Learning Agent")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        episodes = st.slider("Episodes", 100, 2000, 500, step=100)
    with col2:
        alpha = st.slider("Learning Rate (α)", 0.01, 1.0, 0.1, step=0.01)
    with col3:
        gamma = st.slider("Discount Factor (γ)", 0.1, 0.99, 0.9, step=0.01)
    with col4:
        epsilon = st.slider("Exploration Rate (ε)", 0.0, 1.0, 0.2, step=0.01)

    st.info(
        f"**Training Parameters:** α={alpha}, γ={gamma}, ε={epsilon}\n\n"
        "- **α (alpha)**: Learning rate - how quickly the agent updates its knowledge\n"
        "- **γ (gamma)**: Discount factor - importance of future rewards\n"
        "- **ε (epsilon)**: Exploration rate - probability of random actions"
    )

    if st.button("Train Q-Learning Agent", type="primary"):
        with st.spinner("Training agent... This may take a moment."):
            try:
                Q, rewards = q_learning(
                    episodes=episodes, alpha=alpha, gamma=gamma, epsilon=epsilon
                )
                st.success("Training complete!")
                st.session_state.Q = Q

                st.subheader("Training Progress: Rewards per Episode")
                rewards_df = pd.DataFrame({"Episode": range(len(rewards)), "Reward": rewards})
                st.line_chart(rewards_df.set_index("Episode"), height=300)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Reward", f"{np.mean(rewards):.2f}")
                with col2:
                    st.metric("Max Reward", f"{np.max(rewards):.2f}")
                with col3:
                    st.metric("Final Reward", f"{rewards[-1]:.2f}")

                st.subheader("Learned Policy")

                if isinstance(Q, dict):
                    policy_data = []
                    for state, actions in Q.items():
                        best_action = np.argmax(actions)
                        best_reward = actions[best_action]
                        shop, items = state
                        state_str = f"Shop {shop+1}, Items {items}"
                        policy_data.append({
                            "State": state_str,
                            "Best Shop": best_action + 1,
                            "Expected Reward": f"{best_reward:.2f}"
                        })
                    policy_df = pd.DataFrame(policy_data)
                else:
                    best_actions = np.argmax(Q, axis=1)
                    policy_df = pd.DataFrame({
                        "State": [f"State {i+1}" for i in range(len(best_actions))],
                        "Best Shop": best_actions + 1,
                        "Expected Reward": [f"{Q[i, best_actions[i]]:.2f}" for i in range(len(best_actions))]
                    })
                st.dataframe(policy_df, use_container_width=True)

                with st.expander("View Full Q-Table (Advanced)"):
                    display_q_table(Q)
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

with sim_tab:
    st.subheader("Manual Agent Simulation")

    if "state" not in st.session_state:
        st.session_state.state = (0, (0, 0))
        st.session_state.total_reward = 0
        st.session_state.reward_history = []

    shop, items_bought = st.session_state.state

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Shop", shop + 1)
    with col2:
        st.metric("Total Reward", f"{st.session_state.total_reward:.2f}")
    with col3:
        st.metric("Steps Taken", len(st.session_state.reward_history))

    st.write(f"**Items Bought:** {items_bought}")

    all_items_bought = all(items_bought[i] == 1 for i in range(len(items_bought)))
    if all_items_bought:
        st.success("Congratulations! All items have been purchased!")

    st.write("---")
    col1, col2 = st.columns([3, 1])

    with col1:
        action = st.selectbox(
            "Choose next shop to visit:",
            list(range(no_shops)),
            format_func=lambda x: f"Shop {x+1}"
        )

    with col2:
        if st.button("Move to Shop", type="primary", disabled=all_items_bought):
            try:
                new_state, reward = take_action(st.session_state.state, action)
                st.session_state.state = new_state
                st.session_state.total_reward += reward
                st.session_state.reward_history.append(st.session_state.total_reward)

                if reward > 0:
                    st.success(f"Moved to shop {action+1}. Reward earned: {reward:.2f}")
                else:
                    st.warning(f"Moved to shop {action+1}. Reward: {reward:.2f}")
                st.rerun()
            except Exception as e:
                st.error(f"Error taking action: {str(e)}")

    if "Q" in st.session_state:
        st.write("---")
        st.subheader("AI Recommendation")
        try:
            Q = st.session_state.Q
            current_state = st.session_state.state

            if isinstance(Q, dict):
                if current_state in Q:
                    recommended_action = np.argmax(Q[current_state])
                    expected_reward = Q[current_state][recommended_action]
                    st.info(f"AI suggests visiting **Shop {recommended_action+1}** "
                           f"(Expected reward: {expected_reward:.2f})")
                else:
                    st.warning("Current state not found in Q-table")
            else:
                state_idx = shop
                if state_idx < len(Q):
                    recommended_action = np.argmax(Q[state_idx])
                    expected_reward = Q[state_idx, recommended_action]
                    st.info(f"AI suggests visiting **Shop {recommended_action+1}** "
                           f"(Expected reward: {expected_reward:.2f})")
        except Exception as e:
            st.warning(f"Unable to provide AI recommendation: {str(e)}")

    st.write("---")
    st.subheader("Cumulative Reward Over Time")
    if st.session_state.reward_history:
        reward_df = pd.DataFrame({
            "Step": range(len(st.session_state.reward_history)),
            "Cumulative Reward": st.session_state.reward_history
        })
        st.line_chart(reward_df.set_index("Step"), height=250)
    else:
        st.info("No moves made yet. Choose a shop and move to start tracking rewards.")

    st.write("---")
    if st.button("Reset Simulation", type="secondary"):
        st.session_state.state = (0, (0, 0))
        st.session_state.total_reward = 0
        st.session_state.reward_history = []
        st.success("Simulation reset successfully!")
        st.rerun()

with graph_tab:
    st.subheader("Shop Network Visualization")
    st.write("This graph shows the spatial relationship between shops and the distances between them.")
    plot_shop_network(distances)
