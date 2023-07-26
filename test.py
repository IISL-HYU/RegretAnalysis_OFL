import pickle
import matplotlib.pyplot as plt

code = input('Code:')

with open(f"./result_L/clf_FedOMD_{code}.pkl", "rb") as f:
    r1 = pickle.load(f)
    
for i in range(1000):
    print("%.4f" %r1[i], end=' ')
    if (i+1) % 25 == 0:
        print()
