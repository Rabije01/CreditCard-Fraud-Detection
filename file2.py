import numpy as np

x = np.array([1,2,2,2])
y = np.array([1,2,3,4])

addition = x + y
print(f"Addition : {addition}")

scalar_mult = 2 * x
print(f"Scalar multiplication : {scalar_mult}")

dot_products = np.dot(x, y)
print(f"Dot product : {dot_products}")

l1_norm = np.linalg.norm(x, ord=1)
l2_norm = np.linalg.norm(x, ord=2)
inf_norm = np.linalg.norm(x, ord=np.inf)

print(f"L1 norm : {l1_norm}")
print(f"L2 norm : {l2_norm}")
print(f"Infinity norm: {inf_norm}")
