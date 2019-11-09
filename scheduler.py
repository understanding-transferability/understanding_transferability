import numpy as np

def inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    
    return initial_lr * ((1 + gamma * min(1.0, step / float(max_iter))) ** (- power))


def aToBSheduler(step, A, B, gamma=10, max_iter=10000):
    
    ans = A + (2.0 / (1 + np.exp(- gamma * step * 1.0 / max_iter)) - 1.0) * (B - A)
    return float(ans)