import numpy as np

def random_sample_greedy_search(q_table: np.ndarray, states: list, sample_size: int, epsilon: float, seed: int) -> dict:
    rng = np.random.RandomState(seed)
    num_actions = q_table.shape[1]
    k = min(sample_size, num_actions)
    
    actions = []
    q_values = []
    greedy_matches = 0
    
    for state in states:
        state_q = q_table[state]
        true_greedy_action = int(np.argmax(state_q))
        
        if rng.random() < epsilon:
            chosen_action = int(rng.randint(0, num_actions))
        else:
            sampled_actions = np.sort(rng.choice(num_actions, size=k, replace=False))
            best_idx = int(np.argmax(state_q[sampled_actions]))
            chosen_action = int(sampled_actions[best_idx])
            
        actions.append(chosen_action)
        q_values.append(round(float(state_q[chosen_action]), 4))
        
        if chosen_action == true_greedy_action:
            greedy_matches += 1
            
    greedy_match_rate = round(greedy_matches / len(states), 4) if len(states) > 0 else 0.0
    
    return {
        'actions': actions,
        'q_values': q_values,
        'greedy_match_rate': greedy_match_rate
    }