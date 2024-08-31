import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
import matplotlib.animation as animation

# Define the double-exponential disk model
def double_exponential_disk(x, y, z, M_disk, R_disk, z_disk):
    """
    Double-exponential disk model
    """
    r = np.sqrt(x**2 + y**2)
    z_disk_exp = np.exp(-z / z_disk)
    r_disk_exp = np.exp(-r / R_disk)
    return -M_disk / (r_disk_exp * z_disk_exp)

# Define the galaxy model
def galaxy_potential(x, y, z, M, a, b):
    """
    Galaxy model
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    return -M / np.sqrt(r**2 + (a + np.sqrt(b**2 + z**2))**2)

# Define the disk-halo model
def disk_halo_potential(x, y, z, M_disk, M_halo, R_disk, z_disk, a_halo, b_halo):
    """
    Disk-halo model
    """
    disk_pot = double_exponential_disk(x, y, z, M_disk, R_disk, z_disk)
    halo_pot = galaxy_potential(x, y, z, M_halo, a_halo, b_halo)
    return disk_pot + halo_pot

# Define the equations of motion
def equations_of_motion(state, t, M_disk, M_halo, R_disk, z_disk, a_halo, b_halo):
    """
    Equations of motion
    """
    x, y, z, vx, vy, vz = state
    fx = -disk_halo_potential(x, y, z, M_disk, M_halo, R_disk, z_disk, a_halo, b_halo) * x / np.sqrt(x**2 + y**2 + z**2)
    fy = -disk_halo_potential(x, y, z, M_disk, M_halo, R_disk, z_disk, a_halo, b_halo) * y / np.sqrt(x**2 + y**2 + z**2)
    fz = -disk_halo_potential(x, y, z, M_disk, M_halo, R_disk, z_disk, a_halo, b_halo) * z / np.sqrt(x**2 + y**2 + z**2)
    return [vx, vy, vz, fx, fy, fz]

# Initialize the simulation
N = 10000
x = np.random.uniform(-10, 10, N)
y = np.random.uniform(-10, 10, N)
z = np.random.uniform(-10, 10, N)
vx = np.random.uniform(-10, 10, N)
vy = np.random.uniform(-10, 10, N)
vz = np.random.uniform(-10, 10, N)

M_disk = 1e11
M_halo = 1e16
R_disk = 3.0
z_disk = 0.5
a_halo = 10.5
b_halo = 1.0

# Integrate the equations of motion
t = np.linspace(0, 10, 1000)
states = np.zeros((N, len(t), 6))
for i in range(N):
    state0 = np.array([x[i], y[i], z[i], vx[i], vy[i], vz[i]])
state, info = odeint(equations_of_motion, state0, t, args=(M_disk, M_halo, R_disk, z_disk, a_halo, b_halo), full_output=1, rtol=1e-6, atol=1e-9)    
states[i] = state

# Plot the final positions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(states[:, -1, 0], states[:, -1, 1], states[:, -1, 2], c='b', marker='o')
plt.show()

# Plot the trajectories of the particles over time
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(N):
    ax.plot(states[i, :, 0], states[i, :, 1], states[i, :, 2], c='b', alpha=0.1)
plt.show()

# Create an animation of the particles over time
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def animate(i):
    ax.clear()
    ax.scatter(states[:, i, 0], states[:, i, 1], states[:, i, 2], c='b', marker='o')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
print(info)
ani = animation.FuncAnimation(fig, animate, frames=len(t), interval=20)
plt.show()