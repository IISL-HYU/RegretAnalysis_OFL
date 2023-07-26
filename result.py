import pickle
import matplotlib.pyplot  as plt

task = 'clf'

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

# Normal
plt.plot(l1, r1, 'black', label=r'FedOGD')
plt.plot(l4, r4, 'green', label=r'OFedIQ ($\mathdefault{L=1,p=0.251,s=8,b=1767}$)')
plt.plot(l2, r2, 'blue', label=r'OFedAvg ($\mathdefault{p=0.04}$)')
plt.plot(l3, r3, 'red', label=r'FedOMD ($\mathdefault{L=25}$)')


plt.xlabel('time step (t)')

if task == 'clf':
    plt.ylabel('Accuracy (t)')
elif task == 'reg':
    plt.ylabel('MSE (t)')
    #plt.yscale("log")

plt.rcParams["font.family"] = "Times New Roman"
plt.grid()
plt.legend()

plt.savefig(f'./Figures_L/{task}_{code}.png', dpi=200, facecolor="white")
plt.show()

# #Subplot
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# d = 0.7
# kwargs = dict(marker=[(-1, -d), (1, d)], markersize=15, linestyle="none", color='k', clip_on=False)
# ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
# ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
# fig.subplots_adjust(hspace=0.15)

# ax1.set_ylim(0.005, 0.03)
# ax2.set_ylim(0.00225, 0.005)

# ax1.set_yticks([0.005, 0.01, 0.03])
# ax2.set_yticks([0.0023, 0.003])

# ax1.set_ylabel("")
# ax2.set_ylabel("")

# ax1.set_yscale("log")
# ax2.set_yscale("log")

# ax1.spines['bottom'].set_visible(False)
# ax2.spines['top'].set_visible(False)

# fig.text(0.02, 0.50, "MSE (t)", va='center', rotation = 'vertical', fontsize = 10)


# ax1.plot(l1, r1, 'black', label=r'FedOGD')
# ax1.plot(l2, r2, 'red', label=r'OFedAvg ($\mathdefault{p=0.01}$)')
# ax1.plot(l3, r3, 'blue', label=r'FedOMD ($\mathdefault{L=10}$)')
# ax1.plot(l4, r4, 'green', label=r'OFedIQ ($\mathdefault{p=0.086, s=3, b=9}$)')

# ax2.plot(l1, r1, 'black')
# ax2.plot(l2, r2, 'red')
# ax2.plot(l3, r3, 'blue')
# ax2.plot(l4, r4, 'green')
# ax2.set_xlabel('time step (t)')
# ax1.grid()
# ax2.grid()
# ax1.legend()
# ax2.legend()

# plt.show()

