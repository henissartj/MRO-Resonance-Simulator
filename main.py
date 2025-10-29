import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def MRO_equations(t, Y, m, gamma, k):
    """
    System of coupled differential equations based on the damped differential equation:
    Y[0] = x(t)
    Y[1] = dx/dt
    """
    x = Y[0]
    dxdt = Y[1]
    
    dxdtt = -(gamma/m)*dxdt - (k/m)*x  # homogeneous damped motion equation
    
    return [dxdt, dxdtt]

# Model parameters (modifiable)
m = 1.0       # metaphorical mass
gamma = 0.15  # damping coefficient
k = 1.0       # ontogenetic tension constant

# Initial conditions: x(0) = 1 (initial amplitude), dx/dt(0) = 0 (no initial velocity)
Y0 = [1.0, 0.0]

# Time interval for simulation
t_start = 0.0
t_end = 30.0
t_points = 3000
t_eval = np.linspace(t_start, t_end, t_points)

# Numerical resolution with solve_ivp (Runge-Kutta solver order 5(4))
solution = solve_ivp(MRO_equations, [t_start, t_end], Y0, args=(m, gamma, k), t_eval=t_eval, method='RK45')

# Extraction of results
t = solution.t
x = solution.y[0]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t, x, label=r'$x(t)$ - Damped oscillation')
plt.axhline(0, color='black', lw=0.8)
plt.title('Numerical Simulation of the Ontogenetic Resonance Model (MRO)')
plt.xlabel('Time')
plt.ylabel('Amplitude $x(t)$')
plt.grid(True)
plt.legend()
plt.show()
