from sumtree import SumTree
import random
import numpy as np

class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1
    
    PER_b_increment_per_sampling = 0.001
    absolute_error_upper = 1.

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        if max_priority == 0:
            max_priority = absolute_error_upper

        self.tree.add(self._getPriority(max_priority), sample) 

    def sample(self, n):
        batch = []
        # calculate priority segment
        segment = self.tree.total() / n

        #b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n,), dtype=np.float32)
        
        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        #priority_segment = self.tree.total_priority / n       # priority segment
    
        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1
        
        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total()
        max_weight = (p_min * n) ** (-self.PER_b)
          
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)

            sampling_probabilities = p / self.tree.total()
            
            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            is_weights = np.power(n * sampling_probabilities, -self.PER_b)/ max_weight
                                   

            batch.append( (idx, data, is_weights) )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)
