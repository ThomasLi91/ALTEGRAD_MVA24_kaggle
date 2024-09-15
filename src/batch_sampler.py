import numpy as np
import random

def get_miss_ranked_list_of_list(similarity):
    # returns the elements that are miss ranked in a square similarity matrix
    miss_ranked = []
    N_samples = similarity.shape[0]
    for row in range(N_samples):
        miss_ranked_one_text = []
        positive_similarity_pred = similarity[row,row]
        for column in range(N_samples):
            if similarity[row,column] > positive_similarity_pred:
                miss_ranked_one_text.append(column)
        if len(miss_ranked_one_text) > 0:
            miss_ranked_one_text.append(row)
            miss_ranked.append(miss_ranked_one_text)
            
    return miss_ranked



def get_miss_ranked_batch_resized(miss_ranked, batch_size):
    # cuts off miss_ranked_sublists that are too long
    miss_ranked_batch_resized = []
    for miss_ranked_sublist in miss_ranked:
        if len(miss_ranked_sublist) <= batch_size:
            miss_ranked_batch_resized.append(miss_ranked_sublist)
        else:
            group_size = batch_size - 1
            text_idx = miss_ranked_sublist[-1]
            miss_ranked_sublist = miss_ranked_sublist[:-1]
            n_iter = 1 + (len(miss_ranked_sublist) - 1) // group_size
            for i in range(n_iter):
                corrected_miss_ranked_sublist = miss_ranked_sublist[i*group_size:(i+1)*group_size]
                corrected_miss_ranked_sublist.append(text_idx)
                miss_ranked_batch_resized.append(corrected_miss_ranked_sublist)
    return miss_ranked_batch_resized



def pad_sublist(liste, batch_size, n_total_samples):
    # pads the miss_ranked sublists to make them reach batch_size
    size = batch_size - len(liste)
    if size == 0:
        return liste
    else:
        arr = list(set(range(n_total_samples)) - set(liste))
    pad = list(np.random.choice(arr, size=size, replace=False))
    padded_sublist = liste + pad
    return padded_sublist



def list_set_concat(list1, list2):
    return list(set(list1 + list2))



def pad_sublist(liste, batch_size, n_total_samples):
    size = batch_size - len(liste)
    if size == 0:
        return liste
    else:
        arr = list(set(range(n_total_samples)) - set(liste))
    pad = list(np.random.choice(arr, size=size, replace=False))
    padded_sublist = liste + pad
    return padded_sublist



def get_miss_ranked_padded(miss_ranked_batch_resized, batch_size, n_total_samples):
    miss_ranked_padded = []
    n_miss_ranked = len(miss_ranked_batch_resized)
    iter = 0
    corrected_miss_ranked_sublist = []
    while iter < n_miss_ranked:
        concat_miss_rank_sublist = list_set_concat(corrected_miss_ranked_sublist, miss_ranked_batch_resized[iter])
        if len(concat_miss_rank_sublist) <= batch_size:   
            corrected_miss_ranked_sublist = concat_miss_rank_sublist
        else:
            corrected_miss_ranked_sublist = pad_sublist(corrected_miss_ranked_sublist, batch_size, n_total_samples)
            miss_ranked_padded.append(corrected_miss_ranked_sublist)
            corrected_miss_ranked_sublist = miss_ranked_batch_resized[iter]
        iter += 1
    return miss_ranked_padded



def get_miss_ranked_statistics(miss_ranked):
    miss_ranked_sizes = []
    for sublist in miss_ranked:
        miss_ranked_sizes.append(len(sublist))
    miss_ranked_sizes = np.array(miss_ranked_sizes)
    stats_dico = {}
    stats_dico['len_miss_ranked'] = len(miss_ranked_sizes)
    stats_dico['avg_miss_ranked'] = miss_ranked_sizes.mean() - 1
    stats_dico['max_miss_ranked'] = miss_ranked_sizes.max() - 1
    stats_dico['total_miss_ranked_pairs'] = miss_ranked_sizes.sum()
    return stats_dico




def get_batch_sampler(similarity, batch_size):
    # Main function
    n_total_samples = similarity.shape[0]
    miss_ranked = get_miss_ranked_list_of_list(similarity)
    stats_dico = get_miss_ranked_statistics(miss_ranked)
    print(stats_dico)
    random.shuffle(miss_ranked) # shuffle the order
    miss_ranked_batch_resized = get_miss_ranked_batch_resized(miss_ranked, batch_size)
    batch_sampler = get_miss_ranked_padded(miss_ranked_batch_resized, batch_size, n_total_samples)
    regular_batch_sampler = np.random.permutation(n_total_samples)
    regular_batch_sampler = regular_batch_sampler[:n_total_samples - (n_total_samples % batch_size)]
    regular_batch_sampler = regular_batch_sampler.reshape(n_total_samples//batch_size, batch_size).tolist()
    batch_sampler += regular_batch_sampler
    random.shuffle(batch_sampler)
    return batch_sampler, stats_dico