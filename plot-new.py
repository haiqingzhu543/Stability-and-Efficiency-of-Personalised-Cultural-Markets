import numpy as np
import pickle
import matplotlib.pyplot as plt
from market_simulations import market_simulation
import os

length = 299
greedy_eff_stats = np.zeros((len(os.listdir("greedy_metrics_4group")),length))
greedy_ent_stats = np.zeros((len(os.listdir("greedy_metrics_4group")),length))
count = 0
for filename in os.listdir("greedy_metrics_4group"):
    with open("./greedy_metrics_4group/"+filename, "rb") as fp:
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



quality_eff_stats = np.zeros((len(os.listdir("quality_metrics_4group")),length))
quality_ent_stats = np.zeros((len(os.listdir("quality_metrics_4group")),length))
count = 0
for filename in os.listdir("quality_metrics_4group"):
    with open("./quality_metrics_4group/"+filename, "rb") as fp:
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


random_eff_stats = np.zeros((len(os.listdir("random_metrics_4group")),length))
random_ent_stats = np.zeros((len(os.listdir("random_metrics_4group")),length))
count = 0
for filename in os.listdir("random_metrics_4group"):
    with open("./random_metrics_4group/"+filename, "rb") as fp:
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
fig.set_size_inches(10, 7)
ax.plot(np.arange(len(median_eff_greedy))*1000, median_eff_greedy,"b-", label = "Popularity-ranking median (heterogeneous)")
#ax.fill_between(np.arange(length)*1000,min_greedy,max_greedy, alpha=0.5)

ax.plot(np.arange(len(median_eff_quality))*1000, median_eff_quality,"b-.", label = "Quality-ranking median (heterogeneous)")
#ax.fill_between(np.arange(length)*1000,min_quality,max_quality, alpha=0.5)

ax.plot(np.arange(len(median_eff_random))*1000, median_eff_random,"b:", label = "Random-ranking median (heterogeneous)")
#ax.fill_between(np.arange(length)*1000,min_random,max_random, alpha=0.5)
file_greedy = open("greedy_eff_heter.pickle","wb")
file_quality = open("quality_eff_heter.pickle","wb")
file_random = open("random_eff_heter.pickle","wb")
pickle.dump(median_eff_greedy , file_greedy)
pickle.dump(median_eff_quality , file_quality)
pickle.dump(median_eff_random , file_random)


length = 299
greedy_eff_stats = np.zeros((len(os.listdir("greedy_metrics")),length))
greedy_ent_stats = np.zeros((len(os.listdir("greedy_metrics")),length))
count = 0
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

ax.plot(np.arange(len(median_eff_greedy))*1000, median_eff_greedy,"r-", label = "Popularity-ranking median (homogeneous)")

ax.plot(np.arange(len(median_eff_quality))*1000, median_eff_quality,"r-.", label = "Quality-ranking median (homogeneous)")

ax.plot(np.arange(len(median_eff_random))*1000, median_eff_random,"r:", label = "Random-ranking median (homogeneous)")

file_greedy = open("greedy_eff_homo.pickle","wb")
file_quality = open("quality_eff_homo.pickle","wb")
file_random = open("random_eff_homo.pickle","wb")
pickle.dump(median_eff_greedy , file_greedy)
pickle.dump(median_eff_quality , file_quality)
pickle.dump(median_eff_random , file_random)



plt.xlabel("Number of users")
plt.ylabel("Market Efficiency")
plt.legend()
plt.show()
