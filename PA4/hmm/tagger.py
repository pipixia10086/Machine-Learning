import numpy as np
from util import accuracy
from hmm import HMM
from collections import defaultdict


def model_training(train_data, tags):
    vocab = set()
    for line in train_data:
        words = line.words
        for w in words:
            vocab.add(w)
    observations = list(vocab)
    obs_dict = {o: i for i, o in enumerate(observations)}

    state_dict = {}
    for i in range(len(tags)):
        state_dict[tags[i]] = i

    pi_freq = defaultdict(int)
    transition_freq = {}
    emission_freq = {}
    for tag in tags:
        transition_freq[tag] = defaultdict(int)
        emission_freq[tag] = defaultdict(int)

    for line in train_data:
        tag = line.tags
        words = line.words

        pi_freq[tag[0]] += 1
        states_transition = zip(tag, tag[1:])
        for p1, p2 in states_transition:
            transition_freq[p1][p2] += 1

        for i in range(len(tag)):
            emission_freq[tag[i]][words[i]] += 1

    for i in tags:
        if i not in pi_freq:
            pi_freq[i] = 0

    for p1 in tags:
        for p2 in tags:
            if p2 not in transition_freq[p1]:
                transition_freq[p1][p2] = 0

    for p in tags:
        for v in vocab:
            if v not in emission_freq[p]:
                emission_freq[p][v] = 0

    # convert frequency to distribution
    def freq2prob(d):
        prob_dist = {}
        sum_freq = sum(d.values())
        for p, freq in d.items():
            prob_dist[p] = freq / sum_freq
        return prob_dist

    pi = freq2prob(pi_freq)
    pilist = []
    for i in range(len(pi)):
        pilist.append(pi[tags[i]])
    pi = pilist

    transition = {}
    A = np.zeros([len(tags), len(tags)])
    for p, freq_dis in transition_freq.items():
        transition[p] = freq2prob(freq_dis)
    for i in range(len(tags)):
        for j in range(len(tags)):
            A[i][j] = transition[tags[i]][tags[j]]

    emission = {}
    B = np.zeros([len(tags), len(observations)])
    for p, freq_dis in emission_freq.items():
        emission[p] = freq2prob(freq_dis)
    for i in range(len(tags)):
        for j in range(len(observations)):
            B[i][j] = emission[tags[i]][observations[j]]

    model = HMM(pi, A, B, obs_dict, state_dict)
    return model


def sentence_tagging(test_data, model, tags):
    pi, A, B, obs_dict, state_dict = model.pi, model.A, model.B, model.obs_dict, model.state_dict
    tagging = []
    for d in test_data:
        obs = d.words
        for o in obs:
            if o not in obs_dict:
                obs_dict[o] = max(obs_dict.values()) + 1
                tmp = np.array([pow(10, -6)] * len(tags)).reshape(-1, 1)
                B = np.concatenate([B, tmp], axis=1)
        hmm = HMM(pi, A, B, obs_dict, state_dict)
        tagging.append(hmm.viterbi(obs))
    return tagging
