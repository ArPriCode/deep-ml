import numpy as np

def bellman_q_value(states: list, actions: list, transition_probs: dict, rewards: dict, policy: dict, gamma: float, theta: float = 1e-6) -> dict:
    Q = {(s, a): 0.0 for s in states for a in actions}
    
    while True:
        delta = 0.0
        Q_new = {}
        for s in states:
            for a in actions:
                q_val = 0.0
                for s_prime in states:
                    p = transition_probs.get((s, a, s_prime), 0.0)
                    r = rewards.get((s, a, s_prime), 0.0)
                    
                    next_v = sum(policy.get((s_prime, a_prime), 0.0) * Q[(s_prime, a_prime)] for a_prime in actions)
                    q_val += p * (r + gamma * next_v)
                
                Q_new[(s, a)] = q_val
                delta = max(delta, abs(q_val - Q[(s, a)]))
        
        Q = Q_new
        if delta < theta:
            break
            
    return {k: round(v, 4) for k, v in Q.items()}