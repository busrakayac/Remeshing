
from pathlib import Path
import subprocess
import numpy
from nutils.mesh import gmsh
from nutils import function
from nutils.solver import System
from nutils.SI import Length, Density, Viscosity, Velocity
from nutils.expression_v2 import Namespace

L = Length('0.0008m')
W = Length('0.0004m')
R = Length('0.00003m')

rho = Density('10kg/L')
mu = Viscosity('0.00559Pa*s')
U = Velocity('0.745m/s')

root = Path(__file__).parent
meshfile = root / 'crosschannel_empty.msh'

# Generate and retain the background mesh
subprocess.run([
    'gmsh', str(root / 'crosschannel_empty.geo'),
    '-2', '-order', '2', '-format', 'msh2', '-o', str(meshfile)
], check=True)

topo, geom0 = gmsh(meshfile, space='Xbackground')
fl = topo['fluid']

ns = Namespace()
ns.δ = function.eye(2)
ns.x = geom0 * Length('m')
ns.define_for('x', gradient='∇', jacobians=('dV', 'dS'))

ns.ρf = rho
ns.μf = mu
ns.u = fl.field('u', btype='std', degree=2, shape=(2,)) * U
ns.p = fl.field('p', btype='std', degree=1) * mu * U / R
ns.utest = function.replace_arguments(ns.u, 'u:utest') / mu / U**2
ns.ptest = function.replace_arguments(ns.p, 'p:ptest') / mu / U**2

ns.σ_ij = 'μf (∇_j(u_i) + ∇_i(u_j)) - p δ_ij'
ns.uin = U * (1 - (2 * ns.x[1] / W)**2)

# Boundary conditions: walls, opposing inlets, open outlets
sqr = fl.boundary['wall'].integral('u_k u_k dS' @ ns, degree=4) / (R * U**2)
sqr += fl.boundary['inlet1'].integral('((u_0 - uin)^2 + u_1^2) dS' @ ns, degree=4) / (R * U**2)
sqr += fl.boundary['inlet2'].integral('((u_0 + uin)^2 + u_1^2) dS' @ ns, degree=4) / (R * U**2)

cons = System(sqr, trial='u').solve_constraints(droptol=1e-9)

# Stationary fluid residuals
res_stokes = fl.integral('∇_j(utest_i) σ_ij dV' @ ns, degree=4)
res_conv = fl.integral('utest_i ρf ∇_j(u_i) u_j dV' @ ns, degree=4)
res_cont = fl.integral('ptest ∇_i(u_i) dV' @ ns, degree=4)

res = res_stokes + res_conv + res_cont
args = {
    'u': numpy.nan_to_num(cons['u'], nan=0.),
    'p': numpy.zeros(function.arguments_for(res)['p'].shape)
}

# Stokes-to-Navier-Stokes continuation
for alpha in numpy.linspace(0, 1, 11):
    system = System(
        res_stokes + float(alpha) * res_conv + res_cont,
        trial=['u', 'p'], test=['utest', 'ptest']
    )
    args = system.solve(constrain=cons, arguments=args, tol=1e-9)
    print(f'Continuation {alpha:.2f} finished')

# Save FE coefficients and matching physical parameters
numpy.savez_compressed(
    root / 'crossslot_background.npz',
    u=args['u'], p=args['p'],
    U=float(U / 'm/s'),
    rho=float(rho / 'kg/m3'),
    mu=float(mu / 'Pa*s'),
    R=float(R / 'm'),
    L=float(L / 'm'),
    W=float(W / 'm')
)

print('Background flow saved.')
