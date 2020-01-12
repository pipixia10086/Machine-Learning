from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here
        ###################################################

        for i in range(S):
            alpha[i, 0] = self.pi[i] * self.B[i, self.obs_dict[Osequence[0]]]

        for t in range(1, L):
            for j in range(S):
                t1 = self.B[j, self.obs_dict[Osequence[t]]]
                t2 = np.sum([self.A[i, j] * alpha[i, t - 1] for i in range(S)])
                alpha[j, t] = t1 * t2

        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here
        ###################################################
        for i in range(S):
            beta[i, L-1] = 1

        for t in reversed(range(L - 1)):
            for i in range(S):
                beta[i, t] = sum([beta[j, t + 1] * self.A[i, j] *
                                  self.B[j, self.obs_dict[Osequence[t + 1]]] for j in range(S)])

        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here
        ###################################################
        alpha = self.forward(Osequence)
        prob = np.sum(alpha[:, -1])

        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        # Edit here
        ###################################################
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        sequence_prob = self.sequence_prob(Osequence)
        prob = np.multiply(alpha, beta)/sequence_prob

        return prob

    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        # Edit here
        ###################################################
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        denum = self.sequence_prob(Osequence)
        for t in range(L - 1):
            for i in range(S):
                for j in range(S):
                    prob[i, j, t] = alpha[i, t] * beta[j, t + 1] * self.A[i, j] * \
                                    self.B[j, self.obs_dict[Osequence[t + 1]]]/denum

        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        # Q3.3 Edit here
        ###################################################

        S = len(self.pi)
        L = len(Osequence)
        delta = np.zeros([S, L])
        phi = np.zeros([S, L])

        for i in range(S):
            delta[i, 0] = self.pi[i] * self.B[i, self.obs_dict[Osequence[0]]]

        for t in range(1, L):
            for j in range(S):
                tmp = [delta[i, t - 1] * self.A[i, j] for i in range(S)]
                delta[j, t] = max(tmp) * self.B[j, self.obs_dict[Osequence[t]]]
                phi[j, t] = np.argmax(tmp)

        path.append(np.argmax(delta[:, -1]))
        for t in reversed(range(1, L)):
            end = int(path[-1])
            path.append(int(phi[end, t]))

        new_dict = {v: str(k) for k, v in self.state_dict.items()}
        tmp = [0] * L
        for i, p in enumerate(reversed(path)):
            tmp[i] = new_dict[p]
        path = tmp
        return path
