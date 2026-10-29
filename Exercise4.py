# Exercise 4. Workshop 2.


import numpy as np
import matplotlib.pyplot as plt

# 1. Define the ODE function and True Analytical Solution
def f(t, y):
    return (t / 2.0) * (2.0 - y)

def exact_solution(t):
    
    return 2.0 - np.exp(-(t**2) / 4.0)

# 2. Implementation of Fourth-Order Runge-Kutta (RK4) Method
def rk4_method(t_span, y0, h):
   
    t0, tf = t_span
    t_values = np.arange(t0, tf + h, h)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0

    for i in range(len(t_values) - 1):
        t_i = t_values[i]
        y_i = y_values[i]

        k1 = f(t_i, y_i)
        k2 = f(t_i + 0.5 * h, y_i + 0.5 * h * k1)
        k3 = f(t_i + 0.5 * h, y_i + 0.5 * h * k2)
        k4 = f(t_i + h, y_i + h * k3)

        y_values[i + 1] = y_i + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return t_values, y_values

# 3. Implementation of Implicit Trapezoidal Method 
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
step_sizes_rk4 = [1.0, 0.1, 0.01]

print("Error Analysis for RK4 vs Exact Solution ")
plt.figure(figsize=(12, 7))


t_dense = np.linspace(t_span[0], t_span[1], 1000)
plt.plot(t_dense, exact_solution(t_dense), 'k-', label='Exact Analytical Solution', linewidth=2.5)

# Plot RK4 results for step sizes specified in Exercise 4
rk4_colors = ['crimson', 'royalblue', 'mediumseagreen']
rk4_markers = ['o', 's', '^']

for h, color, marker in zip(step_sizes_rk4, rk4_colors, rk4_markers):
    t_rk4, y_rk4 = rk4_method(t_span, y0, h)
    
    # Calculate error
    y_true = exact_solution(t_rk4)
    max_error = np.max(np.abs(y_true - y_rk4))
    print(f"RK4  | Step size h = {h:4.2f} | Max Error = {max_error:.8e}")

    plt.plot(t_rk4, y_rk4, label=f'RK4 (h = {h})', color=color, marker=marker, linestyle='--', alpha=0.85)

# Plot Trapezoidal Method for h = 0.1 for direct comparison with Exercise 2
t_trap, y_trap = trapezoidal_method(t_span, y0, 0.1)
max_error_trap = np.max(np.abs(exact_solution(t_trap) - y_trap))
print(f"TRAP | Step size h = 0.10 | Max Error = {max_error_trap:.8e}")

plt.plot(t_trap, y_trap, 'x:', color='orange', label='Trapezoidal (h = 0.1)', linewidth=2)


plt.title("Exercise 4: RK4 Method Comparison with Exact Solution and Trapezoidal Method", fontsize=13)
plt.xlabel("t", fontsize=12)
plt.ylabel("y(t)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10, loc='best')
plt.tight_layout()

plt.show()