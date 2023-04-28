import numpy as np
from scipy.stats import entropy
import scipy
import matplotlib.pyplot as plt



class market_simulation(object):

    # Rating matrix: An |U| \times |I| matrix containing all ratings

    # untrained_mask: An |U| \times |I| 0/1 matrix, where if (i,j) is not in the training set, the corresponding
    # entry is 1

    # user_group: A list of lists containing the indices of all groups.

    def __init__(self, rating_matrix, untrained_mask, user_groups, r):
        # Initialisation
        print("### Initialising ###")
        self.n_users = rating_matrix.shape[0]
        self.n_items = rating_matrix.shape[1]
        self.n_groups = len(user_groups)
        self.qualities = np.zeros((self.n_groups, self.n_items))
        self.appeals = np.zeros((self.n_groups, self.n_items))
        self.group_weights = np.zeros(self.n_groups)
        self.r = r
        self.initial_choices = np.random.randint(2, 4, size=self.n_items)
        print("Setting: "+ str(self.n_groups) + " group(s)")
        # Divide the ratings into group ratings
        for group_id in range(self.n_groups):
            self.group_weights[group_id] = len(user_groups[group_id])
            # Split the observed and unobserved components
            group_ratings = rating_matrix[user_groups[group_id], :]
            group_unobserved = untrained_mask[user_groups[group_id], :] * group_ratings
            group_observed = (1-untrained_mask[user_groups[group_id], :] )* group_ratings

            # quality is the (normalised) average unobserved values in the user_group
            # a.sum(0) / (a != 0).sum(0)
            self.qualities[group_id] = (group_unobserved.sum(0)/(group_unobserved!= 0).sum(0))/5
            # appeal is the (normalised) average observed values in the user_group
            self.appeals[group_id] = (group_observed.sum(0)/(group_observed!= 0).sum(0))/5

        # Normalise the group weights
        self.group_weights = self.group_weights / np.sum(self.group_weights)
        # Deal with nan entries
        homo_qualities = np.load("qualities_homo.npy")
        homo_appeals = np.load("appeals_homo.npy")
        homo_qualities = np.repeat(homo_qualities,self.n_groups,axis=0)
        homo_appeals = np.repeat(homo_appeals,self.n_groups,axis=0)
        self.qualities[np.isnan(self.qualities)] = homo_qualities[np.isnan(self.qualities)]
        self.appeals[np.isnan(self.appeals)] = homo_appeals[np.isnan(self.appeals)]

        # ??? Choose only unobserved items
        self.untrained = np.zeros((self.n_groups,self.n_items))
        for i in range(len(user_groups)):
            nonzero_inds = np.nonzero(np.sum(untrained_mask[user_groups[i]],axis=0))[0]
            self.untrained[i,nonzero_inds] = 1
        self.qualities = self.qualities*self.untrained
        self.appeals = self.appeals*self.untrained

        # Uncomment to save quality and appeal data
        #np.save("qualities.npy",self.qualities)
        #np.save("appeals.npy",self.appeals)

    def run(self, num_iter, num_iter_per_record=1000):
        metric_records = dict()

        metric_records["market_shares"] = []
        metric_records["obj"] = []
        metric_records["eff"] = []
        metric_records["entropy"] = []

        acc_choices = self.initial_choices
        num_success_perchase = 0
        print("### Running ###")
        # Main loop for simulation
        for iter in range(num_iter):
            # Compute the market share
            market_share = acc_choices / np.sum(acc_choices)

            # Sample a user group
            sampled_group_id = np.random.choice(self.n_groups, p=self.group_weights)

            # Compute the trial-prob-dist
            trial_prob_dist = self.appeals[sampled_group_id] * (market_share ** self.r)

            trial_prob_dist = trial_prob_dist / np.sum(trial_prob_dist)

            # Generate trial
            sampled_item_id = np.random.choice(self.n_items, p=trial_prob_dist)
            flag_success = np.random.binomial(1, self.qualities[sampled_group_id, sampled_item_id])

            if flag_success == 1:
                acc_choices[sampled_item_id] += 1
                num_success_perchase += 1

            # Generated evaluation metrics
            if iter % num_iter_per_record == 0 and iter != 0:
                # Compute
                metric_records["market_shares"].append(market_share)
                metric_records["obj"].append(
                    np.prod(np.sum(self.appeals * self.qualities * (market_share ** self.r), axis=1)))
                metric_records["eff"].append(num_success_perchase / iter)
                metric_records["entropy"].append(entropy(market_share))


        return metric_records

    # Greedy ranking strategy
    def run_greedy_ranking(self, num_iter, num_iter_per_record=10):
        # Ranking function

        metric_records = dict()

        metric_records["market_shares"] = []
        metric_records["obj"] = []
        metric_records["eff"] = []
        metric_records["entropy"] = []

        # Generate ranking factors
        from scipy import interpolate
        # Ranking function

        ranking_factors_r = [0.83, 0.75, 0.69, 0.62, 0.58,
                                                         0.48, 0.44, 0.40, 0.37, 0.35,
                                                         0.338, 0.321, 0.317, 0.3106, 0.2751,
                                                         0.2549, 0.2514, 0.2325, 0.2251, 0.2242,
                                                         0.2150, 0.1904, 0.1841, 0.1818, 0.1701,
                                                         0.1658, 0.1537, 0.1364, 0.1308, 0.1267,
                                                         0.1243, 0.1236, 0.1184, 0.1067, 0.1002,
                                                         0.1038, 0.1019, 0.1048, 0.1061, 0.1142,
                                                         0.1260, 0.1316, 0.1484, 0.1504, 0.1553,
                                                         0.1699, 0.1727, 0.1783, 0.1885, 0.2206]
        #plt.bar(np.arange(50)+1,ranking_factors_r)
        #plt.xlabel("Rank")
        #plt.ylabel("Factor")
        #plt.show()
        # Truncate
        ranking_factors = np.zeros(self.n_items)
        ranking_factors[:50] = ranking_factors_r

        num_success_perchase = 0
        acc_choices = self.initial_choices.copy()

        # Append the initial state
        market_share = acc_choices / np.sum(acc_choices)
        metric_records["market_shares"].append(market_share)
        metric_records["obj"].append(
            np.prod(np.sum(self.appeals * self.qualities * (market_share ** self.r), axis=1)))
        metric_records["eff"].append(0.5)
        metric_records["entropy"].append(entropy(market_share))
        print(metric_records)
        print("### Running ###")
        # Main loop for simulation
        for iter in range(num_iter):
            # Compute the market share
            market_share = acc_choices / np.sum(acc_choices)
            # Sample a user group
            sampled_group_id = np.random.choice(self.n_groups, p=self.group_weights)

            # Compute the trial-prob-dist
            _,rank = np.unique(np.argsort(market_share), return_index=True)
            trial_prob_dist = ranking_factors[::-1][rank] * self.appeals[sampled_group_id] * (
                        market_share ** self.r)
            trial_prob_dist = trial_prob_dist / np.sum(trial_prob_dist)


            # Generate trial
            sampled_item_id = np.random.choice(self.n_items, p=trial_prob_dist)
            flag_success = np.random.binomial(1, self.qualities[sampled_group_id, sampled_item_id])

            if flag_success == 1:
                acc_choices[sampled_item_id] += 1
                num_success_perchase += 1

            # Generated evaluation metrics
            if iter % num_iter_per_record == 0 and iter != 0:
                # Compute
                metric_records["market_shares"].append(market_share)
                metric_records["obj"].append(
                    np.prod(np.sum(self.appeals * self.qualities * (market_share ** self.r), axis=1)))
                metric_records["eff"].append(num_success_perchase / iter)
                metric_records["entropy"].append(entropy(market_share))

        return metric_records

    # Random ranking strategy
    def run_random_ranking(self, num_iter, num_iter_per_record=10):
        # Ranking function

        metric_records = dict()

        metric_records["market_shares"] = []
        metric_records["obj"] = []
        metric_records["eff"] = []
        metric_records["entropy"] = []
        # Generate ranking factors
        from scipy import interpolate
        # Ranking function
        ranking_factors_r = [0.83, 0.75, 0.69, 0.62, 0.58,
                                                         0.48, 0.44, 0.40, 0.37, 0.35,
                                                         0.338, 0.321, 0.317, 0.3106, 0.2751,
                                                         0.2549, 0.2514, 0.2325, 0.2251, 0.2242,
                                                         0.2150, 0.1904, 0.1841, 0.1818, 0.1701,
                                                         0.1658, 0.1537, 0.1364, 0.1308, 0.1267,
                                                         0.1243, 0.1236, 0.1184, 0.1067, 0.1002,
                                                         0.1038, 0.1019, 0.1048, 0.1061, 0.1142,
                                                         0.1260, 0.1316, 0.1484, 0.1504, 0.1553,
                                                         0.1699, 0.1727, 0.1783, 0.1885, 0.2206]

        # Truncate
        ranking_factors = np.zeros(self.n_items)
        ranking_factors[:50] = ranking_factors_r
        num_success_perchase = 0

        acc_choices = self.initial_choices.copy()
        # Append the initial state
        market_share = acc_choices / np.sum(acc_choices)
        metric_records["market_shares"].append(market_share)
        metric_records["obj"].append(
            np.prod(np.sum(self.appeals * self.qualities * (market_share ** self.r), axis=1)))
        metric_records["eff"].append(0.5)
        metric_records["entropy"].append(entropy(market_share))
        print(metric_records)
        print("### Running ###")
        # Main loop for simulation
        for iter in range(num_iter):
            # Compute the market share
            market_share = acc_choices / np.sum(acc_choices)

            # Sample a user group
            sampled_group_id = np.random.choice(self.n_groups, p=self.group_weights)

            np.random.shuffle(ranking_factors)

            # Compute the trial-prob-dist
            trial_prob_dist = ranking_factors * self.appeals[sampled_group_id] * (
                    market_share ** self.r)
            trial_prob_dist = trial_prob_dist / np.sum(trial_prob_dist)

            # Generate trial
            sampled_item_id = np.random.choice(self.n_items, p=trial_prob_dist)
            flag_success = np.random.binomial(1, self.qualities[sampled_group_id, sampled_item_id])

            if flag_success == 1:
                acc_choices[sampled_item_id] += 1
                num_success_perchase += 1

            # Generated evaluation metrics
            if iter % num_iter_per_record == 0 and iter != 0:
                # Compute
                metric_records["market_shares"].append(market_share)
                metric_records["obj"].append(
                    np.prod(np.sum(self.appeals * self.qualities * (market_share ** self.r), axis=1)))
                metric_records["eff"].append(num_success_perchase / iter)
                metric_records["entropy"].append(entropy(market_share))

        return metric_records

    # Random ranking strategy
    def run_quality_ranking(self, num_iter, num_iter_per_record=10):
        # Ranking function

        metric_records = dict()

        metric_records["market_shares"] = []
        metric_records["obj"] = []
        metric_records["eff"] = []
        metric_records["entropy"] = []

        # Generate ranking factors
        from scipy import interpolate
        # Ranking function
        ranking_factors_r = [0.83, 0.75, 0.69, 0.62, 0.58,
                                                         0.48, 0.44, 0.40, 0.37, 0.35,
                                                         0.338, 0.321, 0.317, 0.3106, 0.2751,
                                                         0.2549, 0.2514, 0.2325, 0.2251, 0.2242,
                                                         0.2150, 0.1904, 0.1841, 0.1818, 0.1701,
                                                         0.1658, 0.1537, 0.1364, 0.1308, 0.1267,
                                                         0.1243, 0.1236, 0.1184, 0.1067, 0.1002,
                                                         0.1038, 0.1019, 0.1048, 0.1061, 0.1142,
                                                         0.1260, 0.1316, 0.1484, 0.1504, 0.1553,
                                                         0.1699, 0.1727, 0.1783, 0.1885, 0.2206]

        # Truncate
        ranking_factors = np.zeros(self.n_items)
        ranking_factors[:50] = ranking_factors_r

        num_success_perchase = 0

        acc_choices = self.initial_choices.copy()



        # Append the initial state
        market_share = acc_choices / np.sum(acc_choices)
        metric_records["market_shares"].append(market_share)
        metric_records["obj"].append(
            np.prod(np.sum(self.appeals * self.qualities * (market_share ** self.r), axis=1)))
        metric_records["eff"].append(0.5)
        metric_records["entropy"].append(entropy(market_share))
        print(metric_records)
        print("### Running ###")
        # Main loop for simulation
        for iter in range(num_iter):
            # Compute the market share
            market_share = acc_choices / np.sum(acc_choices)

            # Sample a user group
            sampled_group_id = np.random.choice(self.n_groups, p=self.group_weights)
            _, rank = np.unique(np.argsort(self.qualities[sampled_group_id, :]), return_index=True)

            # Compute the trial-prob-dist
            trial_prob_dist = ranking_factors[::-1][rank] * self.appeals[
                sampled_group_id,:] * (market_share ** self.r)
            trial_prob_dist = trial_prob_dist / np.sum(trial_prob_dist)

            # Generate trial
            sampled_item_id = np.random.choice(self.n_items, p=trial_prob_dist)
            flag_success = np.random.binomial(1, self.qualities[sampled_group_id, sampled_item_id])



            if flag_success == 1:
                acc_choices[sampled_item_id] += 1
                num_success_perchase += 1

            # Generated evaluation metrics
            if iter % num_iter_per_record == 0 and iter != 0:
                # Compute
                metric_records["market_shares"].append(market_share)
                metric_records["obj"].append(
                    np.prod(np.sum(self.appeals * self.qualities * (market_share ** self.r), axis=1)))
                metric_records["eff"].append(num_success_perchase / iter)
                metric_records["entropy"].append(entropy(market_share))


        return metric_records


