import numpy as np
import pickle
import matplotlib.pyplot as plt
from market_simulations import market_simulation
import os



# Length of each data
length = 300

# Legends of the plot
legs = ["hetero-quality","hetero-popularity","hetero-random"]



def compute_avg_efficiency(folder_name,length):
    """
    Compute the median efficiency of the given path
    """
    acc_eff = np.zeros(length)
    count = 0
    num_total_files = len(os.listdir(folder_name))
    res = np.zeros((num_total_files,length))
    for filename in os.listdir(folder_name):
        with open("./"+ folder_name + "/"+ filename, "rb") as fp:
            metrics_greedy = pickle.load(fp)
            res[count,:] = metrics_greedy["eff"]
            count += 1
    ans = np.median(res,axis=0)
    return ans

def compute_avg_entropy(folder_name,length):
    """
    Compute the median entropy of the given path
    """
    acc_eff = np.zeros(length)
    count = 0
    num_total_files = len(os.listdir(folder_name))
    res = np.zeros((num_total_files,length))
    for filename in os.listdir(folder_name):
        with open("./"+ folder_name + "/"+ filename, "rb") as fp:
            metrics_greedy = pickle.load(fp)
            res[count,:] = metrics_greedy["entropy"]
            count += 1
    ans = np.median(res,axis=0)
    return ans

def compute_avg_efficiency_with_error_bar(folder_name,length):
    """
    Compute the median efficiency with the first and third quantiles of the given path
    """
    acc_eff = np.zeros(length)
    count = 0
    num_total_files = len(os.listdir(folder_name))
    res = np.zeros((num_total_files,length))
    for filename in os.listdir(folder_name):
        with open("./"+ folder_name + "/"+ filename, "rb") as fp:
            metrics_greedy = pickle.load(fp)
            res[count,:] = metrics_greedy["eff"]
            count += 1
    ans = np.median(res,axis=0)
    return np.quantile(res,0.25,axis= 0),ans, np.quantile(res,0.75,axis=0)

def compute_avg_entropy_with_error_bar(folder_name,length):
    """
    Compute the median entropy with the first and third quantiles of the given path
    """
    acc_eff = np.zeros(length)
    count = 0
    num_total_files = len(os.listdir(folder_name))
    res = np.zeros((num_total_files,length))
    for filename in os.listdir(folder_name):
        with open("./"+ folder_name + "/"+ filename, "rb") as fp:
            metrics_greedy = pickle.load(fp)
            res[count,:] = metrics_greedy["entropy"]
            count += 1
    ans = np.median(res,axis=0)
    return np.quantile(res,0.25,axis= 0),ans, np.quantile(res,0.75,axis=0)


# Plotting routine for figure 2 left
plt.plot((np.arange(299)+1)*1000,compute_avg_efficiency("trial_quality1",300)[1:],'g--,', label = "homo-quality",markevery=20)
plt.plot((np.arange(299)+1)*1000,compute_avg_efficiency("trial_greedy1",300)[1:], 'b--,', label = "homo-popularity",markevery=20)
plt.plot((np.arange(299)+1)*1000,compute_avg_efficiency("trial_random1",300)[1:], 'r--,', label = "homo-random",markevery=20)
plt.plot((np.arange(299)+1)*1000,compute_avg_efficiency("trial_quality4",300)[1:], 'g:^', label = "hetero-quality",markevery=20)
plt.plot((np.arange(299)+1)*1000,compute_avg_efficiency("trial_greedy4",300)[1:],'b:^', label = "hetero-popularity",markevery=20)
plt.plot((np.arange(299)+1)*1000,compute_avg_efficiency("trial_random4",300)[1:], 'r:^', label = "hetero-random",markevery=20)

plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Efficiency (# purchases / # users)")
plt.ylim([0.65,0.9])
plt.show()




# Plotting rountine for figure 2 right

bbox = dict(boxstyle ="round", fc ="0.95", edgecolor ='none')
arrowprops = dict(
    arrowstyle = "->",
    connectionstyle = "angle, angleA = 0, angleB = 90,\
    rad = 10")

xerr_qual = [compute_avg_entropy_with_error_bar("trial_quality4",300)[1][1:] - compute_avg_entropy_with_error_bar("trial_quality4",300)[0][1:],compute_avg_entropy_with_error_bar("trial_quality4",300)[2][1:] - compute_avg_entropy_with_error_bar("trial_quality4",300)[1][1:]]
yerr_qual = [compute_avg_efficiency_with_error_bar("trial_quality4",300)[1][1:] - compute_avg_efficiency_with_error_bar("trial_quality4",300)[0][1:],compute_avg_efficiency_with_error_bar("trial_quality4",300)[2][1:] - compute_avg_efficiency_with_error_bar("trial_quality4",300)[1][1:]]
plt.errorbar(compute_avg_entropy("trial_quality4",300)[1:],compute_avg_efficiency("trial_quality4",300)[1:],yerr=yerr_qual,xerr=xerr_qual, errorevery = 99,markevery = 99, fmt = "g:^", label = "hetero-quality")
plt.annotate("iter=1000",(compute_avg_entropy("trial_quality4",300)[1],compute_avg_efficiency("trial_quality4",300)[1]-0.006),(compute_avg_entropy("trial_quality4",300)[1],compute_avg_efficiency("trial_quality4",300)[1]-0.03),bbox = bbox,arrowprops = arrowprops, c = 'g')
plt.annotate("iter=300000",(compute_avg_entropy("trial_quality4",300)[-1],compute_avg_efficiency("trial_quality4",300)[-1]-0.006),(compute_avg_entropy("trial_quality4",300)[-1],compute_avg_efficiency("trial_quality4",300)[-1]-0.03),bbox = bbox,arrowprops = arrowprops, c = 'g')



xerr_qual = [compute_avg_entropy_with_error_bar("trial_greedy4",300)[1][1:] - compute_avg_entropy_with_error_bar("trial_greedy4",300)[0][1:],compute_avg_entropy_with_error_bar("trial_greedy4",300)[2][1:] - compute_avg_entropy_with_error_bar("trial_greedy4",300)[1][1:]]
yerr_qual = [compute_avg_efficiency_with_error_bar("trial_greedy4",300)[1][1:] - compute_avg_efficiency_with_error_bar("trial_greedy4",300)[0][1:],compute_avg_efficiency_with_error_bar("trial_greedy4",300)[2][1:] - compute_avg_efficiency_with_error_bar("trial_greedy4",300)[1][1:]]
plt.errorbar(compute_avg_entropy("trial_greedy4",300)[1:],compute_avg_efficiency("trial_greedy4",300)[1:],yerr=yerr_qual,xerr=xerr_qual, errorevery = 99,markevery = 99, fmt = "b:^", label = "hetero-popularity")
plt.annotate("iter=1000",(compute_avg_entropy("trial_greedy4",300)[1],compute_avg_efficiency("trial_greedy4",300)[1]+0.02),(compute_avg_entropy("trial_greedy4",300)[1]-1,compute_avg_efficiency("trial_greedy4",300)[1]+0.03),bbox = bbox,arrowprops = arrowprops, c = 'b')
plt.annotate("iter=300000",(compute_avg_entropy("trial_greedy4",300)[-1],compute_avg_efficiency("trial_greedy4",300)[-1]+0.02),(compute_avg_entropy("trial_greedy4",300)[-1],compute_avg_efficiency("trial_greedy4",300)[-1]+0.04),bbox = bbox,arrowprops = arrowprops, c = 'b')


xerr_qual = [compute_avg_entropy_with_error_bar("trial_random4",300)[1][1:] - compute_avg_entropy_with_error_bar("trial_random4",300)[0][1:],compute_avg_entropy_with_error_bar("trial_random4",300)[2][1:] - compute_avg_entropy_with_error_bar("trial_random4",300)[1][1:]]
yerr_qual = [compute_avg_efficiency_with_error_bar("trial_random4",300)[1][1:] - compute_avg_efficiency_with_error_bar("trial_random4",300)[0][1:],compute_avg_efficiency_with_error_bar("trial_random4",300)[2][1:] - compute_avg_efficiency_with_error_bar("trial_random4",300)[1][1:]]
plt.errorbar(compute_avg_entropy("trial_random4",300)[1:],compute_avg_efficiency("trial_random4",300)[1:],yerr=yerr_qual,xerr=xerr_qual, errorevery = 99,markevery = 99, fmt = "r:^", label = "hetero-random")
plt.annotate("iter=1000",(compute_avg_entropy("trial_random4",300)[1],compute_avg_efficiency("trial_random4",300)[1] - 0.01),(compute_avg_entropy("trial_random4",300)[1]-1,compute_avg_efficiency("trial_random4",300)[1]-0.03),bbox = bbox, arrowprops = arrowprops, c = 'r')
plt.annotate("iter=300000",(compute_avg_entropy("trial_random4",300)[-1],compute_avg_efficiency("trial_random4",300)[-1]+ 0.01),(compute_avg_entropy("trial_random4",300)[-1],compute_avg_efficiency("trial_random4",300)[-1]+0.03),bbox = bbox, arrowprops = arrowprops, c = 'r')


plt.legend()
plt.ylim([0.65,0.9])
plt.xlabel("Entropy (in nats)")
plt.ylabel("Efficiency (# purchases / # users)")
plt.show()