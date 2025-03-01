import numpy as np

#chatGPT
def main():
    gamma = 0.9
    theta = 1e-6  # Soglia di convergenza
    
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
        delta = 0  # Cambiamento massimo in v
        for i in range(len(s)):
            if s[i] == 'Sleep':
                continue  # Stato terminale, il valore rimane zero
            
            v_new = r[i] + gamma * np.sum(P[i] * v)  # Equazione di Bellman
            delta = max(delta, abs(v_new - v[i]))  # Calcolo della variazione
            v[i] = v_new
        
        if delta < theta:  # Convergenza raggiunta
            break
    
    for i in range(len(s)):
        print(f'{s[i]}: {v[i]:.6f}')

if __name__ == '__main__':
    main()
