# Exercise 2. Workshop 2.

import numpy as np
import matplotlib.pyplot as plt

# 1. Define the Ordinary Differential Equation (ODE) and Exact Solution
def f(t, y):
 
    return (t / 2.0) * (2.0 - y)

def exact_solution(t):
  
    return 2.0 - np.exp(-(t**2) / 4.0)

# 2. Implementation of the Trapezoidal Method
def trapezoidal_method(t_span, y0, h):

    t0, tf = t_span
    t_values = np.arange(t0, tf + h, h)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0

    for i in range(len(t_values) - 1):
        t_curr = t_values[i]
        t_next = t_values[i + 1]
        y_curr = y_values[i]
        
       
        numerator = y_curr + (h / 4.0) * (t_curr * (2.0 - y_curr) + 2.0 * t_next)
        denominator = 1.0 + (h * t_next / 4.0)
        
        y_values[i + 1] = numerator / denominator

    return t_values, y_values


t_span = (0.0, 4.0)
y0 = 1.0
step_sizes = [0.1, 0.01, 0.001]


plt.figure(figsize=(10, 6))


t_dense = np.linspace(t_span[0], t_span[1], 1000)
plt.plot(t_dense, exact_solution(t_dense), 'k-', label='True Analytical Solution', linewidth=2)

# Solve and Plot for each step size h
colors = ['red', 'blue', 'green']
linestyle = ['--', '-.', ':']

for h, color, ls in zip(step_sizes, colors, linestyle):
    t_num, y_num = trapezoidal_method(t_span, y0, h)
    
    
    y_true = exact_solution(t_num)
    max_error = np.max(np.abs(y_true - y_num))
    print(f"Step size h = {h:5f} | Max Error = {max_error:.8e}")
    
    plt.plot(t_num, y_num, label=f'Trapezoidal (h = {h})', color=color, linestyle=ls)


plt.title("Comparison of Trapezoidal Method with Different Step Sizes", fontsize=14)
plt.xlabel("t", fontsize=12)
plt.ylabel("y(t)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11)
plt.tight_layout()


plt.show()