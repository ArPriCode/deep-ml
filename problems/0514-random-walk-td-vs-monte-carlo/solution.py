def random_walk_td_mc(episodes: list, n_states: int, gamma: float = 1.0, alpha: float = 0.1) -> tuple:
    td_values = [0.0] * n_states
    
    for episode in episodes:
        for state, reward, next_state, done in episode:
            if done:
                td_target = reward
            else:
                td_target = reward + gamma * td_values[next_state]
            td_values[state] += alpha * (td_target - td_values[state])
            
    mc_returns = [[] for _ in range(n_states)]
    
    for episode in episodes:
        visited_states = set()
        for t, (state, reward, next_state, done) in enumerate(episode):
            if state not in visited_states:
                visited_states.add(state)
                G = 0.0
                discount = 1.0
                for _, r, _, _ in episode[t:]:
                    G += discount * r
                    discount *= gamma
                mc_returns[state].append(G)
                
    mc_values = [
        sum(returns) / len(returns) if returns else 0.0
        for returns in mc_returns
    ]
    
    return td_values, mc_values