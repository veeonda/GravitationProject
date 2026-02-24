import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def init_nbody(N, bounds=(0, 250), min_dist=20.0, mass_range=(50, 200),
               speed_scale=8.0, seed=9, max_tries=200_000):
    lo, hi = bounds
    rng = np.random.default_rng(seed)
    m = rng.uniform(*mass_range, N)

    pos = np.empty((N, 2), float)
    d2min = min_dist**2
    k = t = 0
    while k < N:
        t += 1
        if t > max_tries:
            raise RuntimeError("Placement failed: reduce N/min_dist or increase bounds.")
        c = rng.uniform(lo, hi, 2)
        if k == 0 or np.all(np.sum((pos[:k] - c)**2, axis=1) >= d2min):
            pos[k] = c
            k += 1

    vel = rng.normal(0.0, speed_scale, (N, 2))
    vel -= (m[:, None] * vel).sum(axis=0) / m.sum()  
    return pos, vel, m

def accel(pos, m, G=150.0, eps=10.0):
    r = pos[None, :, :] - pos[:, None, :]
    d2 = np.sum(r*r, axis=2) + eps**2
    np.fill_diagonal(d2, np.inf)
    inv_d3 = 1.0 / (d2 * np.sqrt(d2))
    return G * np.sum(r * inv_d3[:, :, None] * m[None, :, None], axis=1)

def simulate(pos, vel, m, steps=5000, dt=0.0005, G=150.0, eps=10.0):
    N = pos.shape[0]
    traj = np.empty((steps + 1, N, 2), float)
    traj[0] = pos
    a = accel(pos, m, G, eps)
    for k in range(1, steps + 1):
        pos = pos + vel*dt + 0.5*a*dt*dt
        a2 = accel(pos, m, G, eps)
        vel = vel + 0.5*(a + a2)*dt
        a = a2
        traj[k] = pos
    return traj

N, steps, dt = 15, 10000, 0.0005 # actual input values
pos0, vel0, m = init_nbody(N, bounds=(0, 250), min_dist=20.0, mass_range=(50, 200),
                           speed_scale=8.0, seed=9)
traj = simulate(pos0, vel0, m, steps=steps, dt=dt, G=150.0, eps=10.0)

# steps
stride = 25

frames = range(0, steps + 1, stride)

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(0, 250)
ax.set_ylim(0, 250)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_title("N-body Gravitation Animation")

# Color by mass: light -> heavy
norm = plt.Normalize(m.min(), m.max())
cmap = plt.cm.viridis
colors = cmap(norm(m))   # shape (N,4)

sizes = 3 + 4 * np.sqrt(m / m.max())

# Add a colorbar (static, drawn once)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Mass")

points, trails = [], []

for i in range(N):
    c = colors[i]
    trail, = ax.plot([], [], '-', color=c, lw=1.0, alpha=0.6)
    point, = ax.plot([], [], 'o', color=c, ms=float(sizes[i]))
    trails.append(trail)
    points.append(point)

def init():
    for p, t in zip(points, trails):
        p.set_data([], [])
        t.set_data([], [])
    return points + trails

def update(frame):
    for i in range(N):
        x = traj[:frame+1, i, 0]
        y = traj[:frame+1, i, 1]
        trails[i].set_data(x, y)
        points[i].set_data([x[-1]], [y[-1]])
    return points + trails

ani = FuncAnimation(
    fig,
    update,
    frames=frames,
    init_func=init,
    interval=20,
    blit=True
)

plt.close(fig)

# Save GIF (GitHub-friendly)
ani.save("nbody_animation.gif", writer="pillow", fps=24)


# ------------------------------------------------------------------------#
#                                                                         #
#                         Sun+Earth Plot Animation                        #
#                                                                         #
# ------------------------------------------------------------------------#

#Defining Values
m_earth = 3.003e-6 
m_sun = 1.0
G = 1

pos_sun = [0, 0]
pos_earth = [1, 0]
r = 1 

vel_earth = [0, 1]
vel_sun = [0, 0]

# Initialize arrays for Mass, Position, and Velocity
# Uses float for 64-bit precision
m = np.array([m_sun, m_earth], dtype=float)
pos = np.array([pos_sun, pos_earth], dtype=float)
vel = np.array([vel_sun, vel_earth], dtype=float)

# re-define Sun's initial velocity as calculated value
vel[0] = -(m[1] / m[0]) * vel[1]

def accelerations(pos, m, G=1.0, eps=1e-3):
    """
    pos: (N,2)
    m:   (N,)
    returns acc: (N,2)
    eps: softening term for near 0 distances
    """
    N = pos.shape[0]
    acc = np.zeros_like(pos)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            r_vec = pos[j] - pos[i]                 # vector from i -> j
            r2 = np.dot(r_vec, r_vec) + eps**2      # softened r^2
            inv_r3 = 1.0 / (r2 * np.sqrt(r2))       # 1 / r^3
            acc[i] += G * m[j] * r_vec * inv_r3     # a_i += G m_j r_vec / r^3

    return acc

def step_velocity_verlet(pos, vel, m, dt, G=1.0, eps=1e-3):
    a = accelerations(pos, m, G=G, eps=eps)
    pos_new = pos + vel * dt + 0.5 * a * dt**2 # kinematic equation
    a_new = accelerations(pos_new, m, G=G, eps=eps)
    vel_new = vel + 0.5 * (a + a_new) * dt # kinematic equation
    return pos_new, vel_new

dt = 0.001
steps = 6283  # one orbit is ~2π ≈ 6.28, given unit circle it takes 2π time to make a full loop

traj_earth = np.zeros((steps, 2))
traj_sun = np.zeros((steps, 2))

pos_sim = pos.copy()
vel_sim = vel.copy()

for k in range(steps):
    traj_sun[k] = pos_sim[0]
    traj_earth[k] = pos_sim[1]
    pos_sim, vel_sim = step_velocity_verlet(pos_sim, vel_sim, m, dt, G=G, eps=1e-3)

N = min(len(traj_sun), len(traj_earth))

k = 50 # every k-th step is displayed (50/6283 = (125.66) truncated = 125 total steps shown)
frames = range(0, N, k)

# Sun centered coordinates
traj_sun_rel = traj_sun[:N] - traj_sun[0]
# Define subplots
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 5))

# Calculate the barycenter of the system
r_cm = (m_sun * traj_sun + m_earth * traj_earth) / (m_sun + m_earth)

# LEFT: Earth orbit (animated)
# + Sun path (not animated)
# + Barycenter (not animated)
xE, yE = traj_earth[:N, 0], traj_earth[:N, 1]
xS, yS = traj_sun[:N, 0], traj_sun[:N, 1]

all_x0 = np.concatenate([xE, xS])
all_y0 = np.concatenate([yE, yS])
pad0 = 0.05 * max(all_x0.max() - all_x0.min(), all_y0.max() - all_y0.min())

ax0.set_xlim(all_x0.min() - pad0, all_x0.max() + pad0)
ax0.set_ylim(all_y0.min() - pad0, all_y0.max() + pad0)
ax0.set_aspect('equal', adjustable='box')
ax0.set_title("Earth Orbit")
ax0.set_xlabel(r"$x$")
ax0.set_ylabel(r"$y$")

# Sun path in blue
ax0.plot(xS, yS, '-', color='blue', alpha=0.6, label="Sun path (static)")

# Earth animated point + trail
earth_point0, = ax0.plot([], [], 'o', label="Earth")
earth_trail0, = ax0.plot([], [], '-', alpha=0.6, label="Earth trail")

# Barycenter left plot
cm_point0, = ax0.plot([], [], 'x', color='black', label="Barycenter")
ax0.legend()

# RIGHT: Sun orbit (animated)
# + Barycenter (not animated)
xSr, ySr = traj_sun_rel[:, 0], traj_sun_rel[:, 1]

rangeS = max(xSr.max() - xSr.min(), ySr.max() - ySr.min())
pad1 = 0.20 * rangeS
if rangeS == 0:
    rangeS = 1e-6
    pad1 = 1e-6

ax1.set_xlim(xSr.min() - pad1, xSr.max() + pad1)
ax1.set_ylim(ySr.min() - pad1, ySr.max() + pad1)
ax1.set_aspect('equal', adjustable='box')
ax1.set_title("Sun Motion")
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$y$")

# Animated Sun point + trail
sun_point1, = ax1.plot([], [], 'o', label="Sun")
sun_trail1, = ax1.plot([], [], '-', color='blue', alpha=0.6, label="Sun trail")

# Barycenter right plot
traj_sun_rel = traj_sun - traj_sun[0]
r_cm_rel = r_cm[:N] - traj_sun[0]
cm_point1, = ax1.plot([], [], 'x', color='black', label="Barycenter")

ax1.legend()

def init():
    earth_point0.set_data([], [])
    earth_trail0.set_data([], [])
    sun_point1.set_data([], [])
    sun_trail1.set_data([], [])
    cm_point0.set_data([], [])
    cm_point1.set_data([], [])
    return earth_point0, earth_trail0, sun_point1, sun_trail1, cm_point0, cm_point1

def update(i):
    # Earth (left): animate only Earth trail/point
    earth_point0.set_data([traj_earth[i, 0]], [traj_earth[i, 1]])
    earth_trail0.set_data(traj_earth[:i+1, 0], traj_earth[:i+1, 1])

    # Sun (right): animate point in centered coords (no tracking; axes fixed)
    sun_point1.set_data([traj_sun_rel[i, 0]], [traj_sun_rel[i, 1]])
    sun_trail1.set_data(traj_sun_rel[:i+1, 0], traj_sun_rel[:i+1, 1])

    # full-system barycenter
    cm_point0.set_data([r_cm[i, 0]], [r_cm[i, 1]])

    # sun-centered barycenter
    cm_point1.set_data([r_cm_rel[i, 0]], [r_cm_rel[i, 1]])

    return earth_point0, earth_trail0, sun_point1, sun_trail1, cm_point0, cm_point1

ani = FuncAnimation(fig, update, frames=frames, init_func=init,
                    interval=16, blit=True)

plt.close(fig)
ani.save("SunEarth.gif", writer="pillow", fps=10)