import numpy as np
import pickle
import matplotlib.pyplot as plt
from market_simulations import market_simulation

import datetime
file_time_greedy = "greedy_metrics "+str(np.random.rand())
file_time_random = "random_metrics "+str(np.random.rand())
file_time_quality = "quality_metrics "+str(np.random.rand())
"""
rating_matrix = np.random.rand(5,5)
untrained_mask = rating_matrix > 0.5
splited_indices = [[1,2,3],[0,4]]

"""

rating_matrix = np.load("full_rating_matrix.npy")
untrained_mask = np.load("untrained_mask.npy")
with open("groups", "rb") as fp:
    splited_indices = pickle.load(fp)

"""
np.random.seed(10)
rating_matrix = np.random.rand(5,20)*3
untrained_mask = rating_matrix > 1.5
splited_indices = [[1,2,3],[0,4]]
#splited_indices = [[0,1,2,3,4]]
rating_matrix[1,0] = 5
rating_matrix[2,0] = 5
rating_matrix[3,0] = 5

rating_matrix[0,1] = 5
rating_matrix[4,1] = 5
"""

#### Modify this!!!
for i in range(25):
    file_time_greedy = "greedy_metrics " + str(np.random.rand())
    file_time_random = "random_metrics " + str(np.random.rand())
    file_time_quality = "quality_metrics " + str(np.random.rand())

    simulation = market_simulation(rating_matrix, untrained_mask, splited_indices, 0.5)
    #np.random.seed(1)
    #simulation.qualities = np.random.rand(np.shape(simulation.qualities)[0],np.shape(simulation.qualities)[1])

    #for i in range(simulation.qualities.shape[0]):
    #    plt.scatter(simulation.appeals[i,:],simulation.qualities[i,:])
    #plt.show()

    n_iter = 300000

    print(np.corrcoef(simulation.qualities.flatten(),simulation.appeals.flatten()))
    metrics_greedy = simulation.run_greedy_ranking(n_iter, num_iter_per_record=1000)
    metrics_random = simulation.run_random_ranking(n_iter, num_iter_per_record=1000)
    metrics_quality = simulation.run_quality_ranking(n_iter, num_iter_per_record=1000)


    np.save("m_shares.npy",metrics_quality["market_shares"])
    with open(file_time_greedy , "wb") as fp:
        pickle.dump(metrics_greedy,fp)
    with open(file_time_random, "wb") as fp:
        pickle.dump(metrics_random,fp)
    with open(file_time_quality, "wb") as fp:
        pickle.dump(metrics_quality,fp)




"""
plt.figure(1)
plt.plot(metrics_greedy["eff"], label = "greedy_eff")
plt.plot(metrics_random["eff"], label = "random_eff")
plt.plot(metrics_quality["eff"], label = "quality_eff")
#plt.axhline(y=np.max(simulation.qualities), color='r', linestyle='-')
plt.legend()
plt.show()


plt.figure(2)
plt.plot(metrics_greedy["entropy"], label = "greedy_entropy")
plt.plot(metrics_random["entropy"], label = "random_entropy")
plt.plot(metrics_quality["entropy"], label = "quality_entropy")
plt.legend()
plt.show()

plt.figure(3)
plt.plot(metrics_greedy["entropy"], label = "greedy_entropy")
plt.plot(metrics_random["entropy"], label = "random_entropy")
plt.plot(metrics_quality["entropy"], label = "quality_entropy")
plt.legend()
plt.show()
"""

