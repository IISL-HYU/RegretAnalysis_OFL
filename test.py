import pickle
import matplotlib.pyplot  as plt

task = 'reg'

code = input('Code:')

with open(f"./result_L/{task}_FedOGD_{code}.pkl", "rb") as f:
    r1 = pickle.load(f)
with open(f"./result_L/{task}_OFedAvg_{code}.pkl", "rb") as f:
    r2 = pickle.load(f)
with open(f"./result_L/{task}_FedOMD_{code}.pkl", "rb") as f:
    r3 = pickle.load(f)
with open(f"./result_L/{task}_OFedIQ_{code}.pkl", "rb") as f:
    r4 = pickle.load(f)


l1 = list(range(len(r1))) 
l2 = list(range(len(r2)))
l3 = list(range(len(r3)))
l4 = list(range(len(r4)))

plt.plot(l1, r1, 'black', label=r'FedOGD')
plt.plot(l2, r2, 'red', label=r'OFedAvg ($\mathdefault{p=0.05}$)')
plt.plot(l3, r3, 'blue', label=r'FedOMD ($\mathdefault{L=20}$)')
plt.plot(l4, r4, 'green', label=r'OFedIQ ($\mathdefault{p=0.3004}$)')

plt.xlabel('time step (t)')

if task == 'clf':
    plt.ylabel('Accuracy (t)')
elif task == 'reg':
    plt.ylabel('MSE (t)')
    #plt.yscale("log")

plt.rcParams["font.family"] = "Times New Roman"
plt.grid()
plt.legend()
plt.show()

print("FedOGD : %.5f" %r1[-1])
print("OFedAvg : %.5f" %r2[-1])
print("FedOMD : %.5f" %r3[-1])
print("OFedIQ : %.5f" %r4[-1])