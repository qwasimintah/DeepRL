import numpy as np


def action_space(dim, max_skip=7):

	length = 2 * max_skip * dim +1
	res = np.zeros((length, dim))
	j=1
	for i in range(0, dim):
		for skip in range(1, max_skip+1):
			res[j][i]= -skip
			j+=1
			res[j][i]=skip
			j+=1

	return res.tolist()



print(action_space(1, 4))