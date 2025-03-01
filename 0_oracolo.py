import numpy as np
import random

def main(args=None):
    epoche=10000    
    gamma=0.9
    P=np.array([[0,0.5,0,0,0,0.5,0],
                [0,0,0.8,0,0,0,0.2],
                [0,0,0,0.6,0.4,0,0],
                [0,0,0,0,0,0,1.0],
                [0.2,0.4,0.4,0,0,0,0],
                [0.1,0,0,0,0,0.9,0],
                [0,0,0,0,0,0,1.0]])
    s = np.array(['C1', 'C2', 'C3', 'Pass', 'Pub', 'TK', 'Sleep'])
    r = np.array([-2., -2., -2., 10., 1., -1., 0.])
    v = np.zeros(len(s))

    for state in s:

        for e in range(epoche):
            curr_state=state
            g=r[np.where(s == curr_state)]
            n=0
            while curr_state!='Sleep':
                curr_state=random.choices(s, weights=P[np.where(s == curr_state)].reshape(7,-1), k=1)[0]
                n+=1
                g+=r[np.where(s == curr_state)]*(gamma**n)
            if state != 'Sleep': v[np.where(s == state)]+=g
    
    for i in range(s.shape[0]):
        print(f'{s[i]}: {v[i]/epoche}')


if __name__ == '__main__':
    main()
