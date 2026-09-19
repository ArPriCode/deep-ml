def differential_sarsa(
    transitions: dict, 
    initial_state: str, 
    alpha: float, 
    beta: float, 
    num_steps: int
) -> tuple:
    """
    Differential Sarsa for the average-reward continuing setting.
    """
    # Initialize Q-values for all state-action pairs in transitions to 0.0
    Q = {key: 0.0 for key in transitions.keys()}
    R_bar = 0.0
    
    current_state = initial_state
    
    def select_action(state):
        # Find all valid actions from the given state
        possible_actions = [a for (s, a) in Q.keys() if s == state]
        if not possible_actions:
            return None
        
        # Greedy selection: pick action with maximum Q-value.
        # Break ties by choosing the lexicographically smallest action.
        max_q = max(Q[(state, a)] for a in possible_actions)
        best_actions = [a for a in possible_actions if Q[(state, a)] == max_q]
        return min(best_actions)

    # Choose initial action A using greedy action selection
    current_action = select_action(current_state)

    for _ in range(num_steps):
        if current_action is None or (current_state, current_action) not in transitions:
            break
            
        # Observe reward R and next state S'
        reward, next_state = transitions[(current_state, current_action)]
        
        # Choose next action A' using greedy policy from S'
        next_action = select_action(next_state)
        
        # Get Q(S', A') or 0.0 if next_action is None
        next_q = Q[(next_state, next_action)] if next_action is not None else 0.0
        
        # Calculate Differential TD Error
        delta = reward - R_bar + next_q - Q[(current_state, current_action)]
        
        # Update Average Reward Estimate R_bar
        R_bar += beta * delta
        
        # Update Q-value
        Q[(current_state, current_action)] += alpha * delta
        
        # Step forward
        current_state = next_state
        current_action = next_action

    return Q, R_bar