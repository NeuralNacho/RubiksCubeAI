import numpy as np
import tensorflow as tf
from Cube import *
from environment import *

model = build_dqn(0.001, 12, (6,2,2))
print(model.summary())