import numpy as np

#ChatGPT
def main():
    gamma = 1.0  # Discount factor (if needed)
    theta = 1e-6  # Convergence threshold
    
    P = np.array([[0,0.5,0,0,0,0.5,0],
                  [0,0,0.8,0,0,0,0.2],
                  [0,0,0,0.6,0.4,0,0],
                  [0,0,0,0,0,0,1.0],
                  [0.2,0.4,0.4,0,0,0,0],
                  [0.1,0,0,0,0,0.9,0],
                  [0,0,0,0,0,0,1.0]])
    
    s = np.array(['C1', 'C2', 'C3', 'Pass', 'Pub', 'TK', 'Sleep'])
    r = np.array([-2, -2, -2, 10, 1, -1, 0])
    v = np.zeros(len(s))
    
    while True:
        delta = 0  # Maximum change in v
        for i in range(len(s)):
            if s[i] == 'Sleep':
                continue  # Terminal state, the value remains zero
            
            v_new = r[i] + gamma * np.sum(P[i] * v)  # Bellman equation
            delta = max(delta, abs(v_new - v[i]))  # Compute variation
            v[i] = v_new
        
        if delta < theta:  # Convergence reached
            break
    
    for i in range(len(s)):
        print(f'{s[i]}: {v[i]:.6f}')

if __name__ == '__main__':
    main()
