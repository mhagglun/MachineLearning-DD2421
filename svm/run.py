from svm import SVM
import random, math
import numpy as np


#### Generate data ####
N_A = 20
N_B = 40

seed = 21
sd = 1.2
np.random.seed(seed)
random.seed(seed)

classA = np.concatenate(
    (np.random.rand(N_A,2) * sd + [1.5, 0.5],
    np.random.rand(N_A,2) * sd + [-1.5, 0.5],)
)


classB = np.random.rand(N_B,2) * sd + [0, -0.5] 

# classB = np.concatenate((np.random.rand(N_B,2) * 2 + [0, -0.5],
#                         np.random.rand(N_B,2) * 2 + [2.0, 1.5])
# )

inputs = np.concatenate((classA, classB))
targets = np.concatenate( (np.ones(classA.shape[0]), -np.ones(classB.shape[0])) )

N = inputs.shape[0]     # Number of samples

permute = list(range(N))
random.shuffle(permute)
X = inputs[permute, :]
y = targets[permute]

model = SVM()
model.fit(X, y)
model.visualize()