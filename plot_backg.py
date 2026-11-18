
from pathlib import Path
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from nutils.mesh import gmsh
from nutils import function
from nutils.SI import Velocity

# ------------------------------------------------------------
# Load background solution and mesh
# ------------------------------------------------------------
root = Path(__file__).parent
data = numpy.load(root / 'crossslot_background.npz')
topo, geom = gmsh(root / 'crosschannel_empty.msh', space='Xbackground')

fluid = topo['fluid']
U = Velocity(f'{float(data["U"])}m/s')

u = fluid.field('u', btype='std', degree=2, shape=(2,)) * U
sample = fluid.sample('bezier', 3)

xy = numpy.asarray(function.eval(sample.bind(geom)), dtype=float)
vel = numpy.asarray(function.eval(sample.bind(u), arguments={'u': data['u']}) / 'm/s', dtype=float)
speed = numpy.linalg.norm(vel, axis=1)

L = float(data['L'])
W = float(data['W'])

print(f'Minimum velocity magnitude: {speed.min():.6f} m/s')
print(f'Maximum velocity magnitude: {speed.max():.6f} m/s')

# ------------------------------------------------------------
# Figure 1 - Velocity magnitude
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))

im = ax.tripcolor(xy[:, 0]*1e3, xy[:, 1]*1e3,
                  sample.tri, speed, shading='gouraud', cmap='viridis')

fig.colorbar(im, ax=ax, label='Velocity magnitude [m/s]')
ax.set(xlabel='x [mm]', ylabel='y [mm]', title='Background flow - Velocity magnitude')
ax.set_aspect('equal')
ax.set_xlim(-L*1e3, L*1e3)
ax.set_ylim(-L*1e3, L*1e3)

fig.tight_layout()
fig.savefig(root / 'background_velocity.png', dpi=200)
plt.close(fig)

# ------------------------------------------------------------
# Figure 2 - Streamlines
# ------------------------------------------------------------
x = numpy.linspace(-L, L, 250)
y = numpy.linspace(-L, L, 250)
X, Y = numpy.meshgrid(x, y)

# Remove duplicate sampling coordinates
xy_unique, idx = numpy.unique(xy, axis=0, return_index=True)
vel_unique = vel[idx]

Ux = griddata(xy_unique, vel_unique[:, 0], (X, Y), method='linear')
Uy = griddata(xy_unique, vel_unique[:, 1], (X, Y), method='linear')

# Mask regions outside the cross-slot
inside = ((numpy.abs(X) <= W/2) | (numpy.abs(Y) <= W/2))
Ux[~inside] = numpy.nan
Uy[~inside] = numpy.nan

speed_grid = numpy.sqrt(Ux**2 + Uy**2)

fig, ax = plt.subplots(figsize=(7, 7))

im = ax.pcolormesh(X*1e3, Y*1e3, speed_grid,
                   shading='auto', cmap='viridis')

ax.streamplot(x*1e3, y*1e3, Ux, Uy,
              color='white', density=1.5,
              linewidth=0.7, arrowsize=0.7)

fig.colorbar(im, ax=ax, label='Velocity magnitude [m/s]')
ax.set(xlabel='x [mm]', ylabel='y [mm]', title='Background flow - Streamlines')
ax.set_aspect('equal')
ax.set_xlim(-L*1e3, L*1e3)
ax.set_ylim(-L*1e3, L*1e3)

fig.tight_layout()
fig.savefig(root / 'background_streamlines.png', dpi=200)
plt.close(fig)

print('Background flow figures saved.')
