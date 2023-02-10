import numpy as np
import pickle
import matplotlib.pyplot as plt
from market_simulations import market_simulation
import os

def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))

def pad_zero(a):
    return np.pad(a,(1,0), 'constant', constant_values=0)
length = 300


for filename in os.listdir("trial_quality4"):
    with open("./trial_quality4/"+filename, "rb") as fp:
        metrics_greedy = pickle.load(fp)
        plt.scatter(metrics_greedy['entropy'],metrics_greedy['obj'], label = "hetero-quality")
        break

for filename in os.listdir("trial_greedy4"):
    with open("./trial_greedy4/"+filename, "rb") as fp:
        metrics_greedy = pickle.load(fp)
        plt.scatter(metrics_greedy['entropy'],metrics_greedy['obj'], label = "hetero-greedy")
        break

for filename in os.listdir("trial_random4"):
    with open("./trial_random4/"+filename, "rb") as fp:
        metrics_greedy = pickle.load(fp)
        plt.scatter(metrics_greedy['entropy'],metrics_greedy['obj'], label = "hetero-random")
        break


plt.xlabel("entropy")
plt.ylabel("NSW")
plt.legend()
plt.show()




def compute_avg_efficiency(folder_name,length):
    acc_eff = np.zeros(length)
    count = 0
    for filename in os.listdir(folder_name):
        with open("./"+ folder_name + "/"+ filename, "rb") as fp:
            metrics_greedy = pickle.load(fp)
            #plt.plot(metrics_greedy['market_shares'][-1])
            print(metrics_greedy['market_shares'][-1][[[317,168,407]]])
            acc_eff += metrics_greedy['entropy']
            count += 1
    acc_eff = acc_eff / count

    return acc_eff



plt.plot(compute_avg_efficiency("trial_quality1",300)[1:],'r--', label = "Homogeneous, Quality ranking")
plt.plot(compute_avg_efficiency("trial_greedy1",300)[1:], 'r-.', label = "Homogeneous, Popularity ranking")
plt.plot(compute_avg_efficiency("trial_random1",300)[1:], 'r', label = "Homogeneous, Random ranking")
plt.plot(compute_avg_efficiency("trial_quality4",300)[1:], 'g--', label = "Heterogeneous, Quality ranking")
plt.plot(compute_avg_efficiency("trial_greedy4",300)[1:],'g-.', label = "Heterogeneous, Popularity ranking")
plt.plot(compute_avg_efficiency("trial_random4",300)[1:], 'g', label = "Heterogeneous, Random ranking")

plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Efficiency")
plt.show()

"""
for filename in os.listdir("greedy_metrics"):
    with open("./greedy_metrics/"+filename, "rb") as fp:
        metrics_greedy = pickle.load(fp)
        greedy_eff_stats[count,:] = np.array(metrics_greedy["eff"])
        greedy_ent_stats[count,:] = np.array(metrics_greedy["entropy"])
        count+=1
median_eff_greedy = np.quantile(greedy_eff_stats,0.5, axis=0)
min_greedy = np.quantile(greedy_eff_stats,0.25, axis=0)
max_greedy = np.quantile(greedy_eff_stats,0.75, axis=0)

median_greedy_entro = np.quantile(greedy_ent_stats,0.5, axis=0)
min_greedy_entro = np.quantile(greedy_ent_stats,0.25, axis=0)
max_greedy_entro = np.quantile(greedy_ent_stats,0.75, axis=0)



quality_eff_stats = np.zeros((len(os.listdir("quality_metrics")),length))
quality_ent_stats = np.zeros((len(os.listdir("quality_metrics")),length))
count = 0
for filename in os.listdir("quality_metrics"):
    with open("./quality_metrics/"+filename, "rb") as fp:
        metrics_quality = pickle.load(fp)
        quality_eff_stats[count,:] = np.array(metrics_quality["eff"])
        quality_ent_stats[count,:] = np.array(metrics_quality["entropy"])
        count+=1
median_eff_quality = np.quantile(quality_eff_stats,0.5, axis=0)
min_quality = np.quantile(quality_eff_stats,0.25, axis=0)
max_quality = np.quantile(quality_eff_stats,0.75, axis=0)

median_quality_entro = np.quantile(quality_ent_stats,0.5, axis=0)
min_quality_entro = np.quantile(quality_ent_stats,0.25, axis=0)
max_quality_entro = np.quantile(quality_ent_stats,0.75, axis=0)


random_eff_stats = np.zeros((len(os.listdir("random_metrics")),length))
random_ent_stats = np.zeros((len(os.listdir("random_metrics")),length))
count = 0
for filename in os.listdir("random_metrics"):
    with open("./random_metrics/"+filename, "rb") as fp:
        metrics_random = pickle.load(fp)
        random_eff_stats[count,:] = np.array(metrics_random["eff"])
        random_ent_stats[count,:] = np.array(metrics_random["entropy"])
        count+=1
median_eff_random= np.quantile(random_eff_stats,0.5, axis=0)
min_random = np.quantile(random_eff_stats,0.25, axis=0)
max_random = np.quantile(random_eff_stats,0.75, axis=0)

median_random_entro = np.quantile(random_ent_stats,0.5, axis=0)
min_random_entro = np.quantile(random_ent_stats,0.25, axis=0)
max_random_entro = np.quantile(random_ent_stats,0.75, axis=0)


fig, ax = plt.subplots()
ax.plot(np.arange(len(median_eff_greedy))*1000, median_eff_greedy, label = "Popularity-ranking median")
ax.fill_between(np.arange(length)*1000,min_greedy,max_greedy, alpha=0.5)

ax.plot(np.arange(len(median_eff_quality))*1000, median_eff_quality, label = "Quality-ranking median")
ax.fill_between(np.arange(length)*1000,min_quality,max_quality, alpha=0.5)

ax.plot(np.arange(len(median_eff_random))*1000, median_eff_random, label = "Random-ranking median")
ax.fill_between(np.arange(length)*1000,min_random,max_random, alpha=0.5)

plt.xlabel("Number of users")
plt.ylabel("Market Efficiency")
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(np.arange(len(median_eff_greedy))*1000, median_greedy_entro, label = "Popularity-ranking median")
ax.fill_between(np.arange(length)*1000,min_greedy_entro,max_greedy_entro, alpha=0.5)

ax.plot(np.arange(len(median_quality_entro))*1000, median_quality_entro, label = "Quality-ranking median")
ax.fill_between(np.arange(length)*1000,min_quality_entro,max_quality_entro, alpha=0.5)

ax.plot(np.arange(len(median_random_entro))*1000, median_random_entro, label = "Random-ranking median")
ax.fill_between(np.arange(length)*1000,min_random_entro,max_random_entro, alpha=0.5)

plt.xlabel("Number of users")
plt.ylabel("Entropy")
plt.legend()
plt.show()
"""

"""
mean_eff_greedy = np.array(metrics_greedy_1["entropy"])*0.5 + np.array(metrics_greedy_2["entropy"])*0.5
max_eff_greedy = np.maximum(metrics_greedy_1["entropy"],metrics_greedy_2["entropy"])
min_eff_greedy = np.minimum(metrics_greedy_1["entropy"],metrics_greedy_2["entropy"])

mean_eff_quality= np.array(metrics_quality_1["entropy"])*0.5 + np.array(metrics_quality_2["entropy"])*0.5
max_eff_quality = np.maximum(metrics_quality_1["entropy"],metrics_quality_2["entropy"])
min_eff_quality = np.minimum(metrics_quality_1["entropy"],metrics_quality_2["entropy"])


mean_eff_random= np.array(metrics_random_1["entropy"])*0.5 + np.array(metrics_random_2["entropy"])*0.5
max_eff_random = np.maximum(metrics_random_1["entropy"],metrics_random_2["entropy"])
min_eff_random = np.minimum(metrics_random_1["entropy"],metrics_random_2["entropy"])

fig, ax = plt.subplots()
ax.plot(np.arange(len(mean_eff_greedy))*1000, mean_eff_greedy, label = "greedy_mean")
ax.fill_between(np.arange(len(min_eff_greedy))*1000,min_eff_greedy,max_eff_greedy, alpha=0.5)
ax.plot(np.arange(len(mean_eff_quality))*1000, mean_eff_quality, label = "quality_mean")
ax.fill_between(np.arange(len(min_eff_greedy))*1000,min_eff_quality,max_eff_quality, alpha=0.5)
ax.plot(np.arange(len(mean_eff_random))*1000, mean_eff_random, label = "random_mean")
ax.fill_between(np.arange(len(min_eff_greedy))*1000,min_eff_random,max_eff_random, alpha=0.5)
plt.xlabel("Number of users")
plt.ylabel("Market Entropy")
plt.legend()
plt.show()
"""


"""
def compute_equi(Q,V,w,phi,r):
    a = np.sum(Q*V*phi,axis= 1)/ np.sum(V*phi,axis=1)
    w = np.reshape(w,(1,2))
    print(w*a)
    m = np.sum((np.reshape(w,(2,1))* Q * V * (phi ** r) / np.reshape(np.sum(V*(phi**r), axis= 1), (2,1)))/(np.sum(w*a)),axis=0)
    return np.linalg.norm(m-phi)






s = qualities*appeals
s_sum = np.sum(s,axis=0)

avg_qualities = np.average(qualities,axis=0)
ms= metrics_sep["market_shares"]
m_lim = ms[-1]
print(compute_equi(qualities,appeals,weights,m_lim,r))

dis = []
for i in range(len(ms)):
    dis.append(compute_equi(qualities,appeals,weights,ms[i],r))

plt.figure(1)
plt.title("Gap of equilibrium equation")
plt.plot(dis)
plt.show()

plt.figure("bar",[10,10])
plt.subplot(311)
plt.title("$\sum_{i}v_{ij}q_{ij}$")
plt.bar(np.arange(len(m_lim)),np.sort(s_sum)[::-1])
plt.subplot(312)
plt.title("Limiting market share")
plt.bar(np.arange(len(m_lim)),m_lim[np.argsort(s_sum)[::-1]])
plt.subplot(313)
plt.title("CDF of Limiting market share")
plt.plot(np.arange(len(m_lim)),np.cumsum(m_lim[np.argsort(s_sum)[::-1]]))
plt.show()
"""