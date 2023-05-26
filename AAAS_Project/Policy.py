import numpy as np  
import torch.nn as nn
import math

class Policy(nn.Module):


    def normalize(policy):
        probability_of_any_action_policy = []
        for x in policy:
            t = x.data[0]
            s = t.squeeze(0).numpy().item()
            if np.isnan(s):
                s = 1e-6

            probability_of_any_action_policy.append(s)

        # numpyarray_policy = policy.data[0]

        # dim of numpyarray_policy is equal of num_outputs
        # probability_of_any_action_policy = numpyarray_policy.squeeze(0).numpy()

        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
        # prende un valore dall'array num_outputs con le probabilita probability_of_any_action_policy

        normalized_probability_of_any_action_policy = (probability_of_any_action_policy-np.min(probability_of_any_action_policy))/(np.max(probability_of_any_action_policy)-np.min(probability_of_any_action_policy))

        p = np.asarray(normalized_probability_of_any_action_policy)#.astype('float64')
        p = p / np.sum(p)

        if p[0] == math.nan:
            print("Problema!!")

        return p

    def normalizeZero(policy):
        probability_of_any_action_policy = []
        for x in policy:
            t = x.data[0]
            s = t.squeeze(0).numpy().item()
            if np.isnan(s):
                s = 1e-6

            if s < 0:
                s = 1e-9
            probability_of_any_action_policy.append(s)

        # numpyarray_policy = policy.data[0]

        # dim of numpyarray_policy is equal of num_outputs
        # probability_of_any_action_policy = numpyarray_policy.squeeze(0).numpy()

        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
        # prende un valore dall'array num_outputs con le probabilita probability_of_any_action_policy

        normalized_probability_of_any_action_policy = (probability_of_any_action_policy-np.min(probability_of_any_action_policy))/(np.max(probability_of_any_action_policy)-np.min(probability_of_any_action_policy))

        p = np.asarray(normalized_probability_of_any_action_policy)#.astype('float64')
        p = p / np.sum(p)

        if p[0] == math.nan:
            print("Problema!!")

        return p

    def getActionFromStocasticPolicy(policy):
        numpyarray_policy = policy.data[0]

        # dim of numpyarray_policy is equal of num_outputs
        probability_of_any_action_policy = numpyarray_policy.squeeze(0)

        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
        # prende un valore dall'array num_outputs con le probabilita probability_of_any_action_policy
        probability_of_any_action_policy = (probability_of_any_action_policy-np.min(probability_of_any_action_policy))/(np.max(probability_of_any_action_policy)-np.min(probability_of_any_action_policy))

        p = np.asarray(probability_of_any_action_policy)#.astype('float64')
        p = p / np.sum(p)

        action = np.random.choice(p.size, p=p)

        return action

    def getActionAndValueFromStocasticPolicy(policy):
        numpyarray_policy = policy.data[0]

        # dim of numpyarray_policy is equal of num_outputs
        probability_of_any_action_policy = numpyarray_policy.squeeze(0).numpy()

        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
        # prende un valore dall'array num_outputs con le probabilita probability_of_any_action_policy

        normalized_probability_of_any_action_policy = (probability_of_any_action_policy-np.min(probability_of_any_action_policy))/(np.max(probability_of_any_action_policy)-np.min(probability_of_any_action_policy))

        p = np.asarray(normalized_probability_of_any_action_policy)#.astype('float64')
        p = p / np.sum(p)

        action = np.random.choice(p.size, p=p)

        value = policy.data[0][action]

        return action,value

    def getActionAndValueFromStocasticPolicyV2(policy):

        if False: 
            probability_of_any_action_policy = []
            for x in policy:
                t = x.data[0]
                s = t.squeeze(0).numpy().item()
                probability_of_any_action_policy.append(s)

            # numpyarray_policy = policy.data[0]

            # dim of numpyarray_policy is equal of num_outputs
            # probability_of_any_action_policy = numpyarray_policy.squeeze(0).numpy()

            # https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
            # prende un valore dall'array num_outputs con le probabilita probability_of_any_action_policy

            normalized_probability_of_any_action_policy = (probability_of_any_action_policy-np.min(probability_of_any_action_policy))/(np.max(probability_of_any_action_policy)-np.min(probability_of_any_action_policy))

            p = np.asarray(normalized_probability_of_any_action_policy)#.astype('float64')
            p = p / np.sum(p)

        p = Policy.normalize(policy)
        action = np.random.choice(p.size, p=p)

        value = policy[action]

        return action,value


    def getActionFromGreedyPolicy(policy):
        #numpyarray_policy = policy.data[0]

        probability_of_any_action_policy = []
        for x in policy:
            t = x.data[0]
            s = t.squeeze(0).numpy().item()
            if np.isnan(s):
                s = -1e6
            #if s < 0:
            #    s = -1e6

            probability_of_any_action_policy.append(s*12)

        action = np.argmax(probability_of_any_action_policy).item()

        return action
    def getActionFromEpsilonGreedyPolicy(policy,epsilon):
        numpyarray_policy = policy.data[0]

        action = np.argmax(numpyarray_policy)

        probability_of_any_action_policy = numpyarray_policy.squeeze(0)
        p = np.asarray(probability_of_any_action_policy)#.astype('float64')

        number_of_action = p.size

        for i in range(p.size):
            if i == action:
                numpyarray_policy[i] = 1 - epsilon +    epsilon/number_of_action
            else:
                numpyarray_policy[i] =                  epsilon/number_of_action

        probability_of_any_action_policy = numpyarray_policy.squeeze(0)

        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
        # prende un valore dall'array num_outputs con le probabilita probability_of_any_action_policy

        p = np.asarray(probability_of_any_action_policy)#.astype('float64')
        p = p / np.sum(p)

        action = np.random.choice(number_of_action, p=p)
        
        return action

    def getActionAndValueFromEpsilonGreedyPolicy(policy,epsilon):

        value_of_any_action_policy = []
        for x in policy:
            s = x.item()
            if np.isnan(s):
                s = -1e6
            value_of_any_action_policy.append(s)

        # numpy.argmax
        # Returns the indices of the maximum values along an axis.
        optimal_action_index = np.argmax(value_of_any_action_policy)


        number_of_action = len(value_of_any_action_policy)
        probability_of_any_action_policy = []
        for i in range(0 ,number_of_action,1):
            probability_of_any_action_policy.append(0)
            if i == optimal_action_index:
                probability_of_any_action_policy[i] = 1 - epsilon +    epsilon/number_of_action
            else:
                probability_of_any_action_policy[i] =                  epsilon/number_of_action


        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
        # prende un valore dall'array num_outputs con le probabilita probability_of_any_action_policy


        p = probability_of_any_action_policy / np.sum(probability_of_any_action_policy)

        action = np.random.choice(number_of_action, p=p)
        
        value = policy[action]

        return action,value

    def softmax(x):
        num = np.exp(x)
        den = np.exp(x).sum() + 1e-6
        return num / den

    def getActionAndValueFromSoftmaxPolicy(policy):
       


        # p1 = Policy.normalizeZero(policy)

        value_of_any_action_policy = []
        for x in policy:
            s = x.item()
            #t = x
            #s = t.squeeze(0).numpy().item()
            if np.isnan(s):
                s = -1e6
            #if s < 0:
            #    s = -1e6
            value_of_any_action_policy.append(s - 1)

        p1 = value_of_any_action_policy / np.sum(value_of_any_action_policy)

        p2 = Policy.softmax(p1)
        # la riga sotto � per un baco di numpy , sembra che non riesca a calcolare esattamente la somma a 1
        probability_of_any_action_policy = p2 / np.sum(p2)

        action = np.random.choice(probability_of_any_action_policy.size, p=probability_of_any_action_policy)

        value = policy[action]

        return action,value




