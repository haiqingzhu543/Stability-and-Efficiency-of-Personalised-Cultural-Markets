import numpy as np
import pickle
import matplotlib.pyplot as plt
from market_simulations import market_simulation

import datetime

# Store the metrics
file_time_greedy = "greedy_metrics "+str(np.random.rand())
file_time_random = "random_metrics "+str(np.random.rand())
file_time_quality = "quality_metrics "+str(np.random.rand())

rating_matrix = np.load("full_rating_matrix.npy")
untrained_mask = np.load("untrained_mask.npy")
with open("groups", "rb") as fp:
    splited_indices = pickle.load(fp)


#### Simulation loop, 50 indicate the number of data to be generated
for i in range(50):
    file_time_greedy = "greedy_metrics " + str(np.random.rand())
    file_time_random = "random_metrics " + str(np.random.rand())
    file_time_quality = "quality_metrics " + str(np.random.rand())

    simulation = market_simulation(rating_matrix, untrained_mask, splited_indices, 0.5)

    #plt.scatter(simulation.qualities.flatten(),simulation.appeals.flatten())
    #plt.xlabel("Quality")
    #plt.ylabel("Visibility")
    #plt.show()
    n_iter = 300000
    print(np.corrcoef(simulation.qualities.flatten(),simulation.appeals.flatten()))
    metrics_greedy = simulation.run_greedy_ranking(n_iter, num_iter_per_record=1000)
    metrics_random = simulation.run_random_ranking(n_iter, num_iter_per_record=1000)
    metrics_quality = simulation.run_quality_ranking(n_iter, num_iter_per_record=1000)


    #np.save("m_shares.npy",metrics_quality["market_shares"])
    with open(file_time_greedy , "wb") as fp:
        pickle.dump(metrics_greedy,fp)
    with open(file_time_random, "wb") as fp:
        pickle.dump(metrics_random,fp)
    with open(file_time_quality, "wb") as fp:
        pickle.dump(metrics_quality,fp)

