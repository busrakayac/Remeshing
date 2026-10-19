from nutils import cli, export, function, testing
from nutils.mesh import gmsh
from nutils.numeric import inv
from nutils.solver import System
from nutils.SI import Length, Density, Viscosity, Velocity, Time, Pressure, Acceleration
from nutils.expression_v2 import Namespace
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import treelog as log
import numpy
from scipy.interpolate import LinearNDInterpolator

def particle_shape_from_boundary(xb_m):
    """Return L, B, D, theta from deformed particle-boundary sample points.

    Parameters
    ----------
    xb_m : array-like, shape (npoints, 2)
        Deformed particle-boundary coordinates in meters.

    Returns
    -------
    L_particle_m : float
        Particle extent along the major principal direction.
    B_particle_m : float
        Particle extent along the minor principal direction.
    deformation_index : float
        D = (L - B) / (L + B).
    orientation_rad : float
        Angle of the major principal direction with respect to the x-axis.
    """
    pts = numpy.asarray(xb_m, dtype=float)
    pts = pts[numpy.isfinite(pts).all(axis=1)]

    if pts.shape[0] < 3:
        return numpy.nan, numpy.nan, numpy.nan, numpy.nan

    center = pts.mean(axis=0)
    q = pts - center

    # Principal directions of the boundary point cloud.
    cov = q.T @ q / pts.shape[0]
    eigvals, eigvecs = numpy.linalg.eigh(cov)
    order = numpy.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    # Full particle extent projected on the principal directions.
    proj = q @ eigvecs
    L_particle_m = proj[:, 0].max() - proj[:, 0].min()
    B_particle_m = proj[:, 1].max() - proj[:, 1].min()
    orientation_rad = numpy.arctan2(eigvecs[1, 0], eigvecs[0, 0])

    # Safety check: the first direction should be the major direction.
    if B_particle_m > L_particle_m:
        L_particle_m, B_particle_m = B_particle_m, L_particle_m
        orientation_rad += numpy.pi / 2

    deformation_index = (
        (L_particle_m - B_particle_m) / (L_particle_m + B_particle_m)
        if (L_particle_m + B_particle_m) > 0
        else numpy.nan
    )

    return L_particle_m, B_particle_m, deformation_index, orientation_rad

def order_particle_boundary(xb_m):
    """Order particle-boundary sample points clockwise.

    This is sufficient for the present particle shapes, which remain
    approximately circular/elliptical and are not strongly concave.
    """

    pts = numpy.asarray(xb_m, dtype=float)
    pts = pts[numpy.isfinite(pts).all(axis=1)]

    # Remove duplicate Bezier sample points.
    rounded = numpy.round(pts, decimals=14)
    _, unique_indices = numpy.unique(
        rounded,
        axis=0,
        return_index=True,
    )
    pts = pts[numpy.sort(unique_indices)]

    # Sort points by polar angle around their centroid.
    center = pts.mean(axis=0)

    angles = numpy.arctan2(
        pts[:, 1] - center[1],
        pts[:, 0] - center[0],
    )

    pts = pts[numpy.argsort(angles)]

    # Signed polygon area:
    # positive -> counter-clockwise
    # negative -> clockwise
    x = pts[:, 0]
    y = pts[:, 1]

    signed_area = 0.5 * numpy.sum(
        x * numpy.roll(y, -1)
        - numpy.roll(x, -1) * y
    )

    # The inner boundary of a surface is most conveniently clockwise.
    if signed_area > 0:
        pts = pts[::-1]

    return pts



def _dot2(a, b):
    """Contract two length-2 vectors.

    nutils.SI.Quantity does not implement .sum() or numpy.einsum, so an
    explicit component sum is the only way to contract dimensional arrays.
    """
    return a[0] * b[0] + a[1] * b[1]


def _snap_small(v, scale, rel=1e-12):
    """Snap |v| below rel*scale to exactly 0.

    Boundary coordinates that should be exactly zero often carry
    floating-point residues (e.g. 3.7e-21).  Written out at %.16e these
    become literals that some Gmsh kernels treat as degenerate.
    """
    return 0.0 if abs(v) < rel * scale else v


def write_fluid_remesh_geo(boundary_xy_m, domain, filename, bezier_edges=None):
    """Write a fluid-only Gmsh .geo around the current deformed particle.

    The outer cross-slot geometry is identical to the original.  The particle
    boundary is described by the *current spatial* coordinates ``boundary_xy_m``
    so that the new reference configuration of the fluid mesh coincides with
    the current deformed state.  After loading this mesh the fluid displacement
    field ``d`` must therefore be reset to zero (see ``remesh_fluid``).
    """

    boundary_xy_m = numpy.asarray(boundary_xy_m, dtype=float)

    L = float(domain.channel_length / 'm')
    W = float(domain.channel_height / 'm')
    R = float(domain.cylinder_radius / 'm')

    h_particle = R / 2.5
    h_far      = W / 6.0
    r_grow     = 8.0 * R

    first_point = 1001
    first_line  = 2001

    particle_point_ids = [first_point + i for i in range(len(boundary_xy_m))]
    particle_line_ids  = [first_line  + i for i in range(len(boundary_xy_m))]

    with open(filename, 'w') as f:

        f.write('Mesh.MshFileVersion = 2.2;\n')
        # NOTE: the built-in Gmsh kernel is used deliberately (no
        # SetFactory("OpenCASCADE")).  This model is only ~300x the OCC
        # absolute tolerance (~1e-7 m vs a 3e-5 m particle), so OCC treats
        # near-cardinal boundary points and sub-tolerance coordinate
        # residues as degenerate and refuses to build those Bezier curves.
        # The built-in kernel has no such tolerance machinery and meshes
        # the identical geometry cleanly at the same accuracy.

        # Outer cross-slot
        f.write(f'L = {L:.16e};\n')
        f.write(f'W = {W:.16e};\n\n')

        f.write('Point(1)  = {-L,   -W/2, 0, 1.0};\n')
        f.write('Point(2)  = {-W/2, -W/2, 0, 1.0};\n')
        f.write('Point(3)  = {-W/2, -L,   0, 1.0};\n')
        f.write('Point(4)  = { W/2, -L,   0, 1.0};\n')
        f.write('Point(5)  = { W/2, -W/2, 0, 1.0};\n')
        f.write('Point(6)  = { L,   -W/2, 0, 1.0};\n')
        f.write('Point(7)  = { L,    W/2, 0, 1.0};\n')
        f.write('Point(8)  = { W/2,  W/2, 0, 1.0};\n')
        f.write('Point(9)  = { W/2,  L,   0, 1.0};\n')
        f.write('Point(10) = {-W/2,  L,   0, 1.0};\n')
        f.write('Point(11) = {-W/2,  W/2, 0, 1.0};\n')
        f.write('Point(12) = {-L,    W/2, 0, 1.0};\n\n')

        for i in range(1, 12):
            f.write(f'Line({i}) = {{{i},{i+1}}};\n')
        f.write('Line(12) = {12,1};\n')
        f.write('Curve Loop(100) = {1,2,3,4,5,6,7,8,9,10,11,12};\n\n')

        # Particle boundary points
        for pid, (x, y) in zip(particle_point_ids, boundary_xy_m):
            x = _snap_small(x, L); y = _snap_small(y, L)
            f.write(f'Point({pid}) = {{{x:.16e},{y:.16e},0,{h_particle:.16e}}};\n')
        f.write('\n')

        n = len(particle_point_ids)

        if bezier_edges is None:
            # Fallback: straight segments (low geometric fidelity).
            for i in range(n):
                p1 = particle_point_ids[i]
                p2 = particle_point_ids[(i + 1) % n]
                f.write(f'Line({particle_line_ids[i]}) = {{{p1},{p2}}};\n')
        else:
            # Quadratic Bezier arcs reproducing the solid's P2 edges.
            #
            # For each solid edge with endpoints P0, P1 and midside node Pmid,
            # the quadratic Bezier B(t) = (1-t)^2 P0 + 2t(1-t) C + t^2 P1
            # passes through Pmid at t=1/2 provided
            #     C = 2 Pmid - (P0 + P1)/2.
            # Using these arcs instead of straight chords reduces the
            # geometric error by ~3 orders of magnitude, which is what makes
            # the later Topology.locate of solid quadrature points onto this
            # boundary succeed at a tight tolerance.
            ctrl_first = first_point + n
            for i, (p0_xy, pmid_xy, p1_xy) in enumerate(bezier_edges):
                p0_xy   = numpy.asarray(p0_xy,   dtype=float)
                pmid_xy = numpy.asarray(pmid_xy, dtype=float)
                p1_xy   = numpy.asarray(p1_xy,   dtype=float)
                cx, cy  = 2.0 * pmid_xy - 0.5 * (p0_xy + p1_xy)
                cx = _snap_small(cx, L); cy = _snap_small(cy, L)
                cid = ctrl_first + i
                f.write(f'Point({cid}) = {{{cx:.16e},{cy:.16e},0,{h_particle:.16e}}};\n')
            f.write('\n')
            for i in range(n):
                p1 = particle_point_ids[i]
                p2 = particle_point_ids[(i + 1) % n]
                cid = ctrl_first + i
                f.write(f'Bezier({particle_line_ids[i]}) = {{{p1},{cid},{p2}}};\n')

        line_string = ','.join(str(i) for i in particle_line_ids)

        f.write(f'\nCurve Loop(500) = {{{line_string}}};\n')
        f.write('Plane Surface(400) = {100,500};\n\n')
        f.write(f'Transfinite Curve {{{line_string}}} = 2;\n')
        # Physical groups — same names as the original mesh
        f.write('Physical Surface("fluid") = {400};\n')
        f.write(f'Physical Line("cylinder") = {{{line_string}}};\n')
        f.write('Physical Line("inlet1") = {12};\n')
        f.write('Physical Line("inlet2") = {6};\n')
        f.write('Physical Line("outlet1") = {3};\n')
        f.write('Physical Line("outlet2") = {9};\n')
        f.write('Physical Line("wall") = {1,2,4,5,7,8,10,11};\n\n')

        # Mesh sizing
        f.write('Mesh.MeshSizeFromPoints = 0;\n')
        f.write('Mesh.MeshSizeFromCurvature = 0;\n')
        f.write('Mesh.MeshSizeExtendFromBoundary = 0;\n')
        f.write('Mesh.Algorithm = 6;\n')
        f.write('Mesh.Smoothing = 30;\n')
        f.write('Mesh.Optimize = 1;\n')
        f.write('Mesh.OptimizeNetgen = 1;\n\n')

        # Distance-based refinement around the particle
        f.write('Field[1] = Distance;\n')
        f.write(f'Field[1].CurvesList = {{{line_string}}};\n')
        f.write('\nField[2] = Threshold;\n')
        f.write('Field[2].InField = 1;\n')
        f.write('Field[2].DistMin = 0;\n')
        f.write(f'Field[2].DistMax = {r_grow:.16e};\n')
        f.write(f'Field[2].SizeMin = {h_particle:.16e};\n')
        f.write(f'Field[2].SizeMax = {h_far:.16e};\n')
        f.write('\nBackground Field = 2;\n')


def remesh_fluid(
    current_t_s,
    xb_current_m,
    domain,
    ns,
    solid,
    fluid,
    dynamic,
    args,
    topo,           # original combined topo — solid part never changes
    fluid_state,    # dict tracking the CURRENT fluid fields/topology (see main())
    remesh_index,   # 1-based remesh counter, used to build a unique gmsh space name
):
    """Remesh the fluid domain with the corrected ALE/FSI architecture.

    Architecture:
    - d_s : solid displacement on topo['solid'] (ns.d) — never changes topology
    - d_m : fluid mesh displacement on new_topo['fluid'] (ns.dm) — reset to 0
            at remesh, with Newmark history transferred (not zeroed)
    - Physical velocity u_f = v_m + u_rel is transferred; u_rel^new is
      reconstructed as u_f^old - v_m^new (v_m^new is NOT assumed zero)
    - Gradients computed w.r.t. x_m = X_f^new + d_m (moving ALE geometry)
    - FSI traction coupling uses Topology.locate + Sample.zip to couple the
      persistent solid 'dtest' with the fresh fluid stress field, since the
      two no longer share a finite-element topology after remeshing
    - The interface condition d_m,Γ = P_Γ(d_s - d_s^remesh) is built as a
      reusable (zip-based) least-squares functional, returned to the caller
      so it can be re-solved every timestep (the solid keeps moving after
      the remesh instant, so this constraint must be refreshed every step)
    - fluid_state is sampled generically (not hardcoded to ns.x/ns.v/ns.d),
      so a second (third, ...) remesh samples the CURRENT fluid fields
      correctly, whatever they happen to be named
    """

    log.info(f'=== REMESHING FLUID at t={current_t_s:.6f} s ===')

    # ------------------------------------------------------------------
    # 1. Sample the OLD fluid solution generically via fluid_state
    # ------------------------------------------------------------------
    old_topo_fluid = fluid_state['topo_fluid']
    x_field    = fluid_state['x_field']
    v_field    = fluid_state['v_field']
    a_field    = fluid_state['a_field']
    urel_field = fluid_state['urel_field']
    arel_field = fluid_state['arel_field']
    p_field    = fluid_state['p_field']
    d_field    = fluid_state['d_field']
    d_name     = fluid_state['d_name']
    d0_key     = fluid_state['d0_key']
    u0dt_key   = fluid_state['u0dt_key']
    a0dt2_key  = fluid_state['a0dt2_key']

    old_sample = old_topo_fluid.sample('gauss', 4)

    d0_expr    = function.replace_arguments(d_field, [(d_name, d0_key)])
    u0dt_expr  = function.replace_arguments(d_field, [(d_name, u0dt_key)])
    a0dt2_expr = function.replace_arguments(d_field, [(d_name, a0dt2_key)])
    # urel's Newmark history is always named 'u0'/'a0δt' since the relative
    # velocity field is always (re)created with argument name 'u'.
    u0_expr    = function.replace_arguments(urel_field, [('u', 'u0')])
    a0dt_urel_expr = function.replace_arguments(urel_field, [('u', 'a0δt')])

    (old_x_phys, old_v, old_a, old_urel, old_arel, old_p,
    old_d, old_d0, old_u0dt, old_a0dt2,
    old_u0, old_a0dt_urel) = function.eval(
        [
            old_sample.bind(x_field),

            old_sample.bind(v_field),
            old_sample.bind(a_field),

            old_sample.bind(urel_field),
            old_sample.bind(arel_field),

            old_sample.bind(p_field),

            old_sample.bind(d_field),
            old_sample.bind(d0_expr),
            old_sample.bind(u0dt_expr),
            old_sample.bind(a0dt2_expr),
            old_sample.bind(u0_expr),
            old_sample.bind(a0dt_urel_expr),
        ],
        arguments=args,
    )

    old_uf = old_v + old_urel   # physical fluid velocity u_f = v_m + u_rel

    old_x_m       = numpy.asarray(old_x_phys   / 'm',   dtype=float)
    old_uf_ms     = numpy.asarray(old_uf       / 'm/s', dtype=float)
    old_p_Pa      = numpy.asarray(old_p        / 'Pa',  dtype=float)
    old_vm_ms = numpy.asarray(old_v / 'm/s', dtype=float)
    old_am_ms2 = numpy.asarray(old_a / 'm/s2', dtype=float)
    old_arel_ms2 = numpy.asarray(old_arel / 'm/s2', dtype=float)

    old_d_m = numpy.asarray(old_d / 'm', dtype=float)
    old_d0_m = numpy.asarray(old_d0 / 'm', dtype=float)
    old_u0dt_m = numpy.asarray(old_u0dt / 'm', dtype=float)
    old_a0dt2_m = numpy.asarray(old_a0dt2 / 'm', dtype=float)
    old_urel_ms = numpy.asarray(old_urel / 'm/s', dtype=float)
    old_u0_ms = numpy.asarray(old_u0 / 'm/s', dtype=float)
    old_a0dt_urel_ms = numpy.asarray(old_a0dt_urel / 'm/s', dtype=float)

    # ------------------------------------------------------------------
    # 2. Record the current SOLID boundary state for the P_Γ interface
    #    mapping.  This uses a Gauss sample (not bezier) so it can also
    #    serve as the quadrature rule for the FSI traction coupling and
    #    for the per-timestep d_m,Γ constraint, both built via zip+locate.
    # ------------------------------------------------------------------


    solid_cyl_boundary = topo['solid'].boundary['cylinder']
    solid_cyl_gauss   = solid_cyl_boundary.sample('gauss', 4)
    solid_cyl_pts_now_m = numpy.asarray(
        function.eval(solid_cyl_gauss.bind(ns.x), arguments=args) / 'm', dtype=float)
    d_s_remesh_const_m = numpy.asarray(
        function.eval(solid_cyl_gauss.bind(ns.d), arguments=args) / 'm', dtype=float)

    # Old mesh/solid interface velocity BEFORE remeshing
    old_interface_located = old_topo_fluid.boundary['cylinder'].locate(
        x_field / Length('1m'),
        solid_cyl_pts_now_m,
        tol=float(domain.cylinder_radius / 'm') * 1e-4,
        arguments=args,
        skip_missing=False,
    )

    vm_old_g = function.eval(
        old_interface_located.bind(v_field),
        arguments=args,
    )

    vs_old_g = function.eval(
        solid_cyl_gauss.bind(ns.v),
        arguments=args,
    )

    dv_old = numpy.linalg.norm(
        numpy.asarray((vm_old_g - vs_old_g) / 'm/s'),
        axis=-1,
    )

    vs_old_ref = max(
        numpy.linalg.norm(
            numpy.asarray(vs_old_g / 'm/s'),
            axis=-1,
        ).max(),
        1e-30,
    )

    log.info(
        f'[INTERFACE BEFORE REMESH] ||v_m-v_s|| '
        f'max={dv_old.max():.6e} m/s, '
        f'rel={dv_old.max()/vs_old_ref:.6e}'
    )

    # Extract the solid's quadratic (P2) boundary edges as (P0, Pmid, P1)
    # triples, in element order.  A bezier-3 sample yields exactly these
    # three points per element.  These are converted to quadratic Bezier
    # arcs in the new .geo so the fresh fluid boundary reproduces the solid
    # interface to ~1e-10 m instead of the ~1e-7 m error of straight chords.
    solid_cyl_bez3 = solid_cyl_boundary.sample('bezier', 3)
    solid_cyl_bez3_pts_m = numpy.asarray(
        function.eval(solid_cyl_bez3.bind(ns.x), arguments=args) / 'm', dtype=float)
    solid_p2_edges = [
        solid_cyl_bez3_pts_m[solid_cyl_bez3.getindex(ielem)]
        for ielem in range(solid_cyl_bez3.nelems)
    ]

    # Verify the element order actually traces a closed loop: the end point
    # of each edge must coincide with the start point of the next, and the
    # last must close back onto the first.  Element ordering happens to be
    # traversing for the present topology, but that is an assumption, not a
    # guarantee — if it breaks, the Bezier arcs would be stitched in the
    # wrong order and the fresh boundary would be garbage.  If the check
    # fails we reorder greedily by nearest-neighbour chaining.
    def _edges_form_closed_loop(edges, tol):
        for e_this, e_next in zip(edges, edges[1:] + edges[:1]):
            if numpy.linalg.norm(e_this[2] - e_next[0]) > tol:
                return False
        return True

    _loop_tol = float(domain.cylinder_radius / 'm') * 1e-6
    if not _edges_form_closed_loop(solid_p2_edges, _loop_tol):
        log.info('[REMESH] solid P2 edges are not in traversing order; '
                 'reordering by endpoint chaining.')
        remaining = list(solid_p2_edges)
        ordered   = [remaining.pop(0)]
        while remaining:
            tail = ordered[-1][2]
            j = min(range(len(remaining)),
                    key=lambda k: min(numpy.linalg.norm(remaining[k][0] - tail),
                                      numpy.linalg.norm(remaining[k][2] - tail)))
            e = remaining.pop(j)
            if (numpy.linalg.norm(e[2] - tail) <
                    numpy.linalg.norm(e[0] - tail)):
                e = e[::-1]          # flip edge so it continues the chain
            ordered.append(e)
        solid_p2_edges = ordered
        if not _edges_form_closed_loop(solid_p2_edges, _loop_tol):
            raise RuntimeError(
                'solid cylinder boundary edges do not form a closed loop; '
                'cannot build a consistent fresh fluid boundary.')
        log.info('[REMESH] edge reordering succeeded; closed loop verified.')
    else:
        log.info('[REMESH] solid P2 edge connectivity verified (closed loop).')

    # ------------------------------------------------------------------
    # 3. Generate new fluid mesh, in its own gmsh "space" so that Nutils'
    #    Topology.locate / Sample.zip can combine it with the persistent
    #    solid topology (which lives in the default space 'X').
    # ------------------------------------------------------------------
    # Use the P2 edge START points, in element order, as the .geo boundary
    # vertices.  Element order already traverses the closed loop, so no
    # angular sorting is needed (and sorting would break the edge<->arc
    # correspondence that the Bezier control points rely on).
    xb_ordered = numpy.array([e[0] for e in solid_p2_edges], dtype=float)
    geo_file   = Path(f'remesh_fluid_{current_t_s:.6f}s.geo')
    write_fluid_remesh_geo(xb_ordered, domain, geo_file,
                           bezier_edges=solid_p2_edges)

    fluid_space_name = f'Xfluid{remesh_index}'
    new_topo, new_geom_bare = gmsh(geo_file, dimension=2, order=2, space=fluid_space_name)
    new_geom = new_geom_bare * Length('m')
    log.info(f'New fluid mesh generated from {geo_file} (space={fluid_space_name})')

    # ------------------------------------------------------------------
    # 4. Define new fluid fields on new_topo['fluid']
    # ------------------------------------------------------------------
    new_ns = ns   # reuse namespace object; overwrite fluid-specific entries

    new_ns.dm = new_topo['fluid'].field('dm', btype='std', degree=2, shape=(2,)) * domain.cylinder_radius
    if dynamic:
        new_ns.vm, new_ns.am = dynamic.newmark_defo_named(
            new_ns.dm, name='dm', d0_name='dm0', u0dt_name='vm0δt', a0dt2_name='am0δt2')
    else:
        new_ns.vm = Velocity.wrap(function.zeros((2,)))
        new_ns.am = Acceleration.wrap(function.zeros((2,)))

    new_ns.urel = new_topo['fluid'].field('u', btype='std', degree=2, shape=(2,)) * fluid.velocity
    if dynamic:
        new_ns.arel    = dynamic.newmark_velo(new_ns.urel)
        new_ns.u_i     = 'vm_i + urel_i'
        new_ns.DuDt_i  = 'am_i + arel_i + ∇_j(u_i) urel_j'
    else:
        new_ns.u_i     = 'urel_i'
        new_ns.DuDt_i  = '∇_j(u_i) u_j'
    new_ns.edot_ij = '.5 (∇_j(u_i) + ∇_i(u_j))'

    new_ns.p = (
        new_topo['fluid'].field('p', btype='std', degree=1)
        * fluid.viscosity * fluid.velocity / domain.cylinder_radius
    )

    new_ns.utest = function.replace_arguments(new_ns.urel, 'u:utest') / fluid.viscosity / fluid.velocity**2
    new_ns.ptest = function.replace_arguments(new_ns.p,    'p:ptest') / fluid.viscosity / fluid.velocity**2

    # ------------------------------------------------------------------
    # 5. ns_f: gradients/measures on the MOVING ALE geometry x_m = new_geom + dm
    # ------------------------------------------------------------------
    ns_f = Namespace()
    ns_f.δ      = new_ns.δ
    ns_f.ρf     = new_ns.ρf
    ns_f.μf     = new_ns.μf
    ns_f.xfnew  = new_geom
    ns_f.dm     = new_ns.dm
    ns_f.xm_i   = 'xfnew_i + dm_i'
    ns_f.define_for('xm', gradient='∇', normal='n', jacobians=('dV', 'dS'))
    ns_f.urel   = new_ns.urel
    ns_f.p      = new_ns.p
    ns_f.vm     = new_ns.vm
    ns_f.am     = new_ns.am
    if dynamic:
        ns_f.arel    = new_ns.arel
        ns_f.u_i     = 'vm_i + urel_i'
        ns_f.DuDt_i  = 'am_i + arel_i + ∇_j(u_i) urel_j'
    else:
        ns_f.u_i     = 'urel_i'
        ns_f.DuDt_i  = '∇_j(u_i) u_j'
    ns_f.edot_ij = '.5 (∇_j(u_i) + ∇_i(u_j))'
    ns_f.σ_ij    = 'μf (∇_j(u_i) + ∇_i(u_j)) - p δ_ij'
    ns_f.traction_i = 'σ_ij n_j'   # fluid stress vector, for the FSI coupling
    ns_f.utest   = new_ns.utest
    ns_f.ptest   = new_ns.ptest
    ns_f.dtest   = new_ns.dtest

    # ns_m: mesh-extension namespace, ∇ref w.r.t. the epoch reference new_geom
    ns_m = Namespace()
    ns_m.δ      = new_ns.δ
    ns_m.xfnew  = new_geom
    ns_m.define_for('xfnew', gradient='∇ref', jacobians=('dVref', 'dSref'))
    ns_m.dm     = new_ns.dm
    ns_m.xm_i   = 'xfnew_i + dm_i'
    ns_m.Fm_ij  = '∇ref_j(xm_i)'
    ns_m.Jm     = numpy.linalg.det(ns_m.Fm)
    ns_m.Fminv  = numpy.linalg.inv(ns_m.Fm)
    ns_m.MeshPK_ij = '2 (Fm_ij - Fminv_ji)'
    ns_m.dmtest = function.replace_arguments(new_ns.dm, 'dm:dmtest') / (domain.cylinder_radius**2 * Pressure('1Pa'))

    # ------------------------------------------------------------------
    # 6. Build the residual
    # ------------------------------------------------------------------
    new_res = 0.

    # Solid momentum — unchanged, persistent topology/field
    new_res += topo['solid'].integral(
        '(∇ref_j(dtest_i) P_ij + dtest_i ρs (a_i - g_i)) dVref' @ new_ns,
        degree=4,
    )

    # Mesh extension for d_m (elasticity-type smoothing, epoch reference new_geom)
    new_res += Pressure('1Pa') * new_topo['fluid'].integral(
        '∇ref_j(dmtest_i) MeshPK_ij dVref' @ ns_m, degree=4
    )

    # Fluid momentum and incompressibility, gradients w.r.t. moving x_m
    new_res += new_topo['fluid'].integral(
        '(utest_i ρf DuDt_i + ∇_j(utest_i) σ_ij) dV' @ ns_f, degree=4
    )
    new_res += new_topo['fluid'].integral(
        'ptest ∇_k(u_k) dV' @ ns_f, degree=4
    )

    # ------------------------------------------------------------------
    # FSI coupling: fluid traction acting on the solid boundary.
    #
    # After remeshing, 'd' (solid, on topo['solid']) and 'u','p','dm'
    # (fluid, on new_topo['fluid']) no longer share a common finite-element
    # topology, so the original "volume lift" identity
    #   ∮ d·σ·n dΓ = ∫ (∇d:σ + d·ρDuDt) dV
    # (valid only when d and u share dofs on the SAME mesh) no longer
    # applies.  Instead we impose the coupling directly as a boundary
    # traction term, evaluated at matching physical points via
    # Topology.locate, combined via Sample.zip so that the SOLID's own
    # Gauss quadrature provides the (correct, persistent) weights:
    #   ∮_cylinder dtest_i (σ_ij n_j) dΓ
    # ------------------------------------------------------------------
    # Tolerance must be loose enough to absorb the residual geometric
    # difference between the solid's P2 arcs and the fresh mesh's Bezier
    # arcs (~1e-10 m for a 3e-5 m particle), but far tighter than any
    # physically meaningful length.  We scale it off the particle radius.
    locate_tol_m = float(domain.cylinder_radius / 'm') * 1e-4
    try:
        fluid_traction_located = new_topo['fluid'].boundary['cylinder'].locate(
            new_geom_bare, solid_cyl_pts_now_m, tol=locate_tol_m, skip_missing=False,
        )
    except Exception:
        log.info(
            f'[REMESH] locate failed at tol={locate_tol_m:.3e} m; '
            f'retrying with a looser tolerance.'
        )
        locate_tol_m *= 100.
        fluid_traction_located = new_topo['fluid'].boundary['cylinder'].locate(
            new_geom_bare, solid_cyl_pts_now_m, tol=locate_tol_m, skip_missing=False,
        )
    log.info(f'[REMESH] FSI traction coupling located at tol={locate_tol_m:.3e} m')
    zipped_traction = solid_cyl_gauss.zip(fluid_traction_located)

    new_res += zipped_traction.integral(
        _dot2(new_ns.dtest, ns_f.traction) * new_ns.dS
    )

    # Interface target  P_Γ(d_s - d_s^remesh).
    #
    # d_s_remesh_const_m is a (Nq,2) NumPy array (one value per interface
    # quadrature point).  Subtracting it raw from the (2,) symbolic field
    # would broadcast into a spurious (Nq,2) tensor; asfunction keeps the
    # expression shape (2,).
    d_s_remesh_coeffs = numpy.array(
        args['d'],
        copy=True,
    )

    d_s_remesh_field = function.replace_arguments(
        new_ns.d,
        {'d': d_s_remesh_coeffs},
    )

    dm_interface_target_expr = (
        new_ns.d - d_s_remesh_field
)

    # ------------------------------------------------------------------
    # MONOLITHIC interface kinematic coupling:  d_m,Γ = P_Γ(d_s - d_s^remesh)
    #
    # This is imposed as a Lagrange-multiplier residual INSIDE the Newton
    # system, not as a Dirichlet constraint computed from a predicted d_s.
    # Enforcing it via cons['dm'] would freeze the interface dofs at values
    # derived from the *predicted* solid displacement, giving the staggered
    # scheme
    #     d_m,Γ^{n+1} = P_Γ(d_s,pred^{n+1} - d_s^r)
    # instead of the monolithic
    #     d_m,Γ^{n+1} = P_Γ(d_s^{n+1}    - d_s^r).
    # With the multiplier in the system, Newton updates d_s and d_m,Γ
    # simultaneously and the relation holds at convergence.
    #
    # Multiplier space: the saddle-point system is only inf-sup stable if
    # the multiplier space is POORER than the displacement space.  With
    # degree-2 displacements, a degree-2 multiplier makes the system
    # singular; degree 1 is stable.  (Verified numerically.)
    # ------------------------------------------------------------------
    new_ns.lam = (
        new_topo['fluid'].boundary['cylinder'].field(
            'lam', btype='std', degree=1, shape=(2,))
        * solid.shear_modulus
    )
    # Scaling must render the constraint residual dimensionless:
    #   [lamtest]*[dm]*[dS] = (P/(P L^2)) * L * L = 1     (dS ~ L in 2D)
    # The G*R (rather than G*R^2) scaling leaves it with dimension L and
    # trips an SI DimensionError.
    new_ns.lamtest = function.replace_arguments(
        new_ns.lam, 'lam:lamtest') / (solid.shear_modulus * domain.cylinder_radius**2)

    # Reaction term: +∫ λ·δd_m  ONLY.
    #
    # The multiplier must NOT push back on the solid (no −∫ λ·δd_s term).
    # The mesh-extension problem is artificial numerical machinery; its
    # reaction is not a physical force.  The genuine fluid force on the
    # solid is already supplied by the traction term above.  The coupling
    # is therefore one-directional in the kinematic sense:
    #     d_s  ——> interface constraint ——>  d_m
    new_res += zipped_traction.integral(
        _dot2(ns_m.dmtest, new_ns.lam) * new_ns.dS
    )
    # Constraint term: ∫ δλ·(d_m − (d_s − d_s^remesh))
    new_res += zipped_traction.integral(
        _dot2(new_ns.lamtest, new_ns.dm - dm_interface_target_expr) * new_ns.dS
    )

    # Hydrodynamic force diagnostic (unchanged — pure fluid-side quantity)
    new_hydrodynamic_force = new_topo['fluid'].boundary['cylinder'].integral(
        '-σ_ij n_j dS' @ ns_f, degree=4
    )

    # ------------------------------------------------------------------
    # 7. Rebuild constraints
    #
    #    d_m is fixed to zero only on the OUTER fluid boundary.  The
    #    cylinder (interface) boundary is intentionally NOT constrained
    #    here — its value is d_m,Γ = P_Γ(d_s - d_s^remesh), which must be
    #    refreshed every timestep as the solid keeps moving.  The
    #    ingredients for that per-step update (a zip-based least-squares
    #    functional) are returned to the caller via fluid_state.
    # ------------------------------------------------------------------
    new_cons = {}

    sqr_ds = topo.boundary.integral('d_k d_k dSref' @ new_ns, degree=4) / domain.cylinder_radius**3
    new_cons = System(sqr_ds, trial='d').solve_constraints(droptol=1e-9, constrain=new_cons)

    # Outer (stationary) fluid boundary: dm = 0.  This constraint is FIXED
    # for the whole epoch, so we keep it separately as dm_outer_cons and
    # rebuild the full dm constraint from it every timestep.  Reusing the
    # previous *full* dm constraint instead would silently freeze the
    # interface: Nutils treats finite entries as already-constrained and
    # skips them, giving d_m,Γ^{n+1} = d_m,Γ^n while the solid keeps moving.
    sqr_dm_outer = new_topo['fluid'].boundary['wall,inlet1,inlet2,outlet1,outlet2'].integral(
        'dm_k dm_k dSref' @ ns_m, degree=4
    ) / domain.cylinder_radius**3
    dm_outer_only = System(sqr_dm_outer, trial='dm').solve_constraints(droptol=1e-9)
    dm_outer_cons = dm_outer_only['dm'].copy()

    # The interface value of d_m is NOT constrained here: it is imposed
    # monolithically by the Lagrange-multiplier residual built above, so
    # Newton can update d_s and d_m,Γ together.  Only the stationary outer
    # fluid boundary is a Dirichlet constraint.
    new_cons = dict(new_cons)
    new_cons['dm'] = dm_outer_cons.copy()

    # u (urel): no-slip + inflow on fluid boundary
    ns_bc = Namespace()
    ns_bc.urel  = new_ns.urel
    ns_bc.xfnew = new_geom
    ns_bc.define_for('xfnew', gradient='∇ref', jacobians=('dVref', 'dSref'))
    ns_bc.uin   = fluid.velocity * (1 - (2 * (new_geom[1] / domain.channel_height)
                                          / (domain.channel_height / domain.channel_height)) ** 2)
    sqr_u = (
        new_topo['fluid'].boundary['wall,cylinder'].integral(
            'urel_k urel_k dSref' @ ns_bc, degree=4
        ) / (domain.cylinder_radius * fluid.velocity ** 2)
    )
    sqr_u += (
        new_topo['fluid'].boundary['inlet1'].integral(
            '(urel_0 - uin)^2 dSref' @ ns_bc, degree=4
        ) / (domain.cylinder_radius * fluid.velocity ** 2)
    )
    sqr_u += (
        new_topo['fluid'].boundary['inlet2'].integral(
            '(urel_0 + uin)^2 dSref' @ ns_bc, degree=4
        ) / (domain.cylinder_radius * fluid.velocity ** 2)
    )
    sqr_u += (
        new_topo['fluid'].boundary['inlet1,inlet2'].integral(
            'urel_1^2 dSref' @ ns_bc, degree=4
        ) / (domain.cylinder_radius * fluid.velocity ** 2)
    )
    new_cons = System(sqr_u, trial='u').solve_constraints(droptol=1e-9, constrain=new_cons)
    new_ucons = new_cons['u'].copy()

    # ------------------------------------------------------------------
    # 8. Build new Newton system
    # ------------------------------------------------------------------
    new_system = System(
        new_res,
        trial=['d', 'dm', 'u', 'p', 'lam'],
        test=['dtest', 'dmtest', 'utest', 'ptest', 'lamtest'],
    )

    # ------------------------------------------------------------------
    # 9. Transfer solution to new mesh (physical coordinates, physical
    #    velocity, full Newmark history for both d_m and u_rel)
    # ------------------------------------------------------------------

    # Gauss quadrature is the natural sample for an L2 / least-squares
    # projection (bezier sampling is a visualisation/interpolation device).
    new_dof_sample = new_topo['fluid'].sample('gauss', 4)
    new_dof_x_m    = numpy.asarray(new_dof_sample.eval(new_geom) / 'm', dtype=float)

    # Field transfer: evaluate the OLD fields directly at the NEW sample
    # points by locating those points in the old physical ALE mesh.  This
    # is exact (uses the old FE basis) and, unlike LinearNDInterpolator
    # with fill_value=0, does not silently zero points that fall outside
    # the convex hull of the old Gauss cloud — which happens precisely
    # near the cylinder, where accuracy matters most.
    # Independent of locate_tol_m, which the FSI-coupling block above may
    # have loosened on retry; that must not silently affect field transfer.
    transfer_tol_m = float(domain.cylinder_radius / 'm') * 1e-4
    transfer_via_locate = False
    try:
        # locate must be performed in the OLD mesh's *physical* (deformed
        # ALE) coordinates, because new_dof_x_m are physical positions.
        # x_field is exactly that (= X_old + d_m_old); strip its SI unit
        # and supply args so the displacement part evaluates.
        old_located = old_topo_fluid.locate(
            x_field / Length('1m'), new_dof_x_m,
            tol=transfer_tol_m, arguments=args, skip_missing=False)
        
        (t_uf, t_vm, t_am, t_urel, t_arel, t_p,
        t_dn, t_d0, t_u0dt, t_a0dt2,
        t_u0, t_a0dt_urel) = function.eval([
            old_located.bind(v_field + urel_field), old_located.bind(v_field), old_located.bind(a_field),
            old_located.bind(urel_field), old_located.bind(arel_field), old_located.bind(p_field),
            old_located.bind(d_field), old_located.bind(d0_expr),
            old_located.bind(u0dt_expr), old_located.bind(a0dt2_expr),
            old_located.bind(u0_expr), old_located.bind(a0dt_urel_expr),
        ], arguments=args)


        new_uf_x, new_uf_y = numpy.asarray(t_uf / 'm/s', dtype=float).T
        new_vm_x, new_vm_y = numpy.asarray(t_vm / 'm/s', dtype=float).T
        new_am_x, new_am_y = numpy.asarray(t_am / 'm/s2', dtype=float).T
        new_urel_x, new_urel_y = numpy.asarray(t_urel / 'm/s', dtype=float).T
        new_arel_x, new_arel_y = numpy.asarray(t_arel / 'm/s2', dtype=float).T
        new_p_Pa = numpy.asarray(t_p / 'Pa', dtype=float)
        new_dn_x, new_dn_y = numpy.asarray(t_dn / 'm', dtype=float).T
        new_d0_x, new_d0_y = numpy.asarray(t_d0 / 'm', dtype=float).T
        new_u0dt_x, new_u0dt_y = numpy.asarray(t_u0dt / 'm', dtype=float).T
        new_a0dt2_x, new_a0dt2_y = numpy.asarray(t_a0dt2 / 'm', dtype=float).T
        new_u0_x, new_u0_y = numpy.asarray(t_u0 / 'm/s', dtype=float).T
        new_a0dt_urel_x, new_a0dt_urel_y = numpy.asarray(t_a0dt_urel / 'm/s', dtype=float).T
        
        transfer_via_locate = True
        log.info('[REMESH] field transfer via Topology.locate (exact FE evaluation)')
    except Exception as ex:
        log.info(f'[REMESH] locate-based transfer unavailable ({ex}); '
                 f'falling back to scattered interpolation.')

    if not transfer_via_locate:

        def _I(vals):
            return LinearNDInterpolator(
                old_x_m,
                vals,
                fill_value=0.0,
            )(new_dof_x_m)

        new_uf_x = _I(old_uf_ms[:, 0])
        new_uf_y = _I(old_uf_ms[:, 1])

        new_vm_x = _I(old_vm_ms[:, 0])
        new_vm_y = _I(old_vm_ms[:, 1])

        new_am_x = _I(old_am_ms2[:, 0])
        new_am_y = _I(old_am_ms2[:, 1])

        new_urel_x = _I(old_urel_ms[:, 0])
        new_urel_y = _I(old_urel_ms[:, 1])

        new_arel_x = _I(old_arel_ms2[:, 0])
        new_arel_y = _I(old_arel_ms2[:, 1])

        new_p_Pa = _I(old_p_Pa)

        new_dn_x = _I(old_d_m[:, 0])
        new_dn_y = _I(old_d_m[:, 1])

        new_d0_x = _I(old_d0_m[:, 0])
        new_d0_y = _I(old_d0_m[:, 1])

        new_u0dt_x = _I(old_u0dt_m[:, 0])
        new_u0dt_y = _I(old_u0dt_m[:, 1])

        new_a0dt2_x = _I(old_a0dt2_m[:, 0])
        new_a0dt2_y = _I(old_a0dt2_m[:, 1])

        new_u0_x = _I(old_u0_ms[:, 0])
        new_u0_y = _I(old_u0_ms[:, 1])

        new_a0dt_urel_x = _I(old_a0dt_urel_ms[:, 0])
        new_a0dt_urel_y = _I(old_a0dt_urel_ms[:, 1])


    beta, gamma, dt = dynamic.beta, dynamic.gamma, float(dynamic.timestep / 's')
    # ------------------------------------------------------------------
    # 10. Project all interpolated fields onto new dof arrays via L2 solve
    # ------------------------------------------------------------------
    R_scale   = float(domain.cylinder_radius / 'm')
    u_scale   = float(fluid.velocity / 'm/s')
    p_scale_v = float(fluid.viscosity * fluid.velocity / domain.cylinder_radius / 'Pa')

    def _project_vector(target_x, target_y, scale):
        ns_p = Namespace()
        ns_p.fproj = new_topo['fluid'].field('fproj', btype='std', degree=2, shape=(2,))
        ns_p.ftarget = new_dof_sample.asfunction(
            numpy.stack([target_x / scale, target_y / scale], axis=1).astype(float))
        sqr = new_dof_sample.integral(((ns_p.fproj - ns_p.ftarget) ** 2).sum(-1))
        return System(sqr, trial='fproj').solve(constrain={}, arguments={})['fproj']

    # Synchronize current mesh state at remeshing -----------------------------

    def _project_mesh_state(tx, ty, interface_target):
        ns_h = Namespace()
        ns_h.h = function.replace_arguments(new_ns.dm, [('dm', 'h')])
        ns_h.htarget = new_dof_sample.asfunction(
            numpy.stack([tx / R_scale, ty / R_scale], axis=1).astype(float)
        ) * domain.cylinder_radius

        sqr_gamma = zipped_traction.integral(
            _dot2(ns_h.h - interface_target, ns_h.h - interface_target) * new_ns.dS
        ) / domain.cylinder_radius**3

        hcons = System(sqr_gamma, trial='h').solve_constraints(
            droptol=1e-9,
            constrain={'h': dm_outer_cons.copy()},
            arguments=args,
        )

        sqr_bulk = new_dof_sample.integral(
            (((ns_h.h - ns_h.htarget) / domain.cylinder_radius)**2).sum(-1)
        )

        return System(sqr_bulk, trial='h').solve(
            constrain=hcons, arguments={}
        )['h']


    def _project_mesh_outer(tx, ty):
        ns_h = Namespace()
        ns_h.h = function.replace_arguments(new_ns.dm, [('dm', 'h')])
        ns_h.htarget = new_dof_sample.asfunction(
            numpy.stack([tx / R_scale, ty / R_scale], axis=1).astype(float)
        ) * domain.cylinder_radius

        sqr = new_dof_sample.integral(
            (((ns_h.h - ns_h.htarget) / domain.cylinder_radius)**2).sum(-1)
        )

        return System(sqr, trial='h').solve(
            constrain={'h': dm_outer_cons.copy()},
            arguments={},
        )['h']


    urel_new_dofs = _project_vector(new_urel_x, new_urel_y, u_scale)
    u0_new = _project_vector(new_u0_x, new_u0_y, u_scale)
    a0dt_new = _project_vector(new_a0dt_urel_x, new_a0dt_urel_y, u_scale)

    V_mesh = _project_mesh_state(
        new_vm_x * dt, new_vm_y * dt,
        new_ns.v * dynamic.timestep,
    )

    A_mesh = _project_mesh_state(
        new_am_x * dt**2, new_am_y * dt**2,
        new_ns.a * dynamic.timestep**2,
    )

    dm_n_dofs = numpy.zeros_like(V_mesh)

    A0_mesh = _project_mesh_outer(
        new_a0dt2_x, new_a0dt2_y,
    )

    δA = A_mesh - A0_mesh

    am0dt2_new = A0_mesh
    vm0dt_new = V_mesh - A0_mesh - gamma * δA
    dm0_new = dm_n_dofs - vm0dt_new - .5*A0_mesh - beta*δA

    δaδt2_new = (
        dm_n_dofs - dm0_new - vm0dt_new - .5*am0dt2_new
    ) / beta

    vm_new_dofs = vm0dt_new + am0dt2_new + gamma*δaδt2_new
    am_new_dofs = am0dt2_new + δaδt2_new

    vm_new_dofs_ms = vm_new_dofs * R_scale / dt
    am_new_dofs_ms2 = am_new_dofs * R_scale / dt**2

    log.info(
        f'[CURRENT STATE] V error = '
        f'{numpy.linalg.norm(vm_new_dofs - V_mesh):.6e}'
    )

    log.info(
        f'[CURRENT STATE] A error = '
        f'{numpy.linalg.norm(am_new_dofs - A_mesh):.6e}'
    )

    def _project_scalar(target, scale):
        ns_p = Namespace()
        ns_p.fproj = new_topo['fluid'].field('fproj', btype='std', degree=1)
        ns_p.ftarget = new_dof_sample.asfunction((target / scale).astype(float))
        sqr = new_dof_sample.integral((ns_p.fproj - ns_p.ftarget) ** 2)
        return System(sqr, trial='fproj').solve(constrain={}, arguments={})['fproj']


    uf_proj_dofs = _project_vector(new_uf_x, new_uf_y, u_scale)
    p_proj_dofs  = _project_scalar(new_p_Pa, p_scale_v)

    V_mesh_direct = _project_vector(new_vm_x * dt, new_vm_y * dt, R_scale)
    A_mesh_direct = _project_vector(new_am_x * dt**2, new_am_y * dt**2, R_scale)
    A_rel         = _project_vector(new_arel_x * dt, new_arel_y * dt, u_scale)

    # Relative-acceleration history check
    arel_new_dofs = a0dt_new + (urel_new_dofs - u0_new - a0dt_new) / gamma
    arel_err = numpy.linalg.norm((arel_new_dofs - A_rel) * u_scale / dt)
    log.info(f'[AREL HISTORY CHECK] error={arel_err:.6e} m/s2')

    vm_target = numpy.column_stack((new_vm_x, new_vm_y))
    am_target = numpy.column_stack((new_am_x, new_am_y))

    def check_vm_transfer(label, coeffs):
        val = function.eval(
            new_dof_sample.bind(new_ns.dm / dynamic.timestep),
            arguments={'dm': coeffs},
        )
        val = numpy.asarray(val / 'm/s', dtype=float)
        err = numpy.linalg.norm(val - vm_target, axis=1)
        rms = numpy.sqrt(numpy.mean(err**2))
        ref = numpy.sqrt(numpy.mean(numpy.linalg.norm(vm_target, axis=1)**2))
        log.info(
            f'[VM TRANSFER] {label}: max={err.max():.6e} m/s, '
            f'RMS={rms:.6e} m/s, relRMS={rms/max(ref,1e-30):.6e}'
        )

    def check_am_transfer(label, coeffs):
        val = function.eval(
            new_dof_sample.bind(new_ns.dm / dynamic.timestep**2),
            arguments={'dm': coeffs},
        )
        val = numpy.asarray(val / 'm/s2', dtype=float)
        err = numpy.linalg.norm(val - am_target, axis=1)
        rms = numpy.sqrt(numpy.mean(err**2))
        ref = numpy.sqrt(numpy.mean(numpy.linalg.norm(am_target, axis=1)**2))
        log.info(
            f'[AM TRANSFER] {label}: max={err.max():.6e} m/s2, '
            f'RMS={rms:.6e} m/s2, relRMS={rms/max(ref,1e-30):.6e}'
        )

    check_vm_transfer('direct L2 projection', V_mesh_direct)
    check_am_transfer('direct L2 projection', A_mesh_direct)

    check_vm_transfer('synchronized current state', V_mesh)
    check_am_transfer('synchronized current state', A_mesh)

    # Direct projection with stationary outer-boundary correction
    outer_mask = numpy.isfinite(dm_outer_cons)

    V_mesh_direct_outer = V_mesh_direct.copy()
    A_mesh_direct_outer = A_mesh_direct.copy()

    V_mesh_direct_outer[outer_mask] = 0.
    A_mesh_direct_outer[outer_mask] = 0.

    check_vm_transfer('direct L2 projection + outer BC', V_mesh_direct_outer)
    check_am_transfer('direct L2 projection + outer BC', A_mesh_direct_outer)

    # ------------------------------------------------------------------
    # Check interface BEFORE applying the interface correction
    # ------------------------------------------------------------------

    ns_check = Namespace()

    ns_check.h = function.replace_arguments(new_ns.dm, [('dm', 'h')])
    vm_gamma, vs_gamma = function.eval(
        [zipped_traction.bind(ns_check.h / dynamic.timestep),
        zipped_traction.bind(new_ns.v)],
        arguments=dict(args, h=vm_new_dofs),
    )

    am_gamma, as_gamma = function.eval(
        [zipped_traction.bind(ns_check.h / dynamic.timestep**2),
        zipped_traction.bind(new_ns.a)],
        arguments=dict(args, h=am_new_dofs),
    )
    dv = numpy.linalg.norm(numpy.asarray((vm_gamma - vs_gamma) / 'm/s'), axis=-1)
    da = numpy.linalg.norm(numpy.asarray((am_gamma - as_gamma) / 'm/s2'), axis=-1)

    vs_ref = max(numpy.linalg.norm(numpy.asarray(vs_gamma / 'm/s'), axis=-1).max(), 1e-30)
    as_ref = max(numpy.linalg.norm(numpy.asarray(as_gamma / 'm/s2'), axis=-1).max(), 1e-30)

    log.info(f'[INTERFACE AT REMESH] ||a_m-a_s|| max={da.max():.6e} m/s2, rel={da.max()/as_ref:.6e}')
    log.info(f'[INTERFACE AT REMESH] ||v_m-v_s|| max={dv.max():.6e} m/s, rel={dv.max()/vs_ref:.6e}')

    # ns_hist = Namespace()
    # ns_hist.h = function.replace_arguments(new_ns.dm, [('dm', 'h')])

    # current_interface_targets = {
    #     'V_mesh': new_ns.v * dynamic.timestep,
    #     'A_mesh': new_ns.a * dynamic.timestep**2,
    # }

    # for key, target in current_interface_targets.items():

    #     diff_h = ns_hist.h - target

    #     sqr_h = zipped_traction.integral(
    #         _dot2(diff_h, diff_h) * new_ns.dS
    #     ) / domain.cylinder_radius**3

    #     hcons = System(
    #         sqr_h,
    #         trial='h',
    #     ).solve_constraints(
    #         droptol=1e-9,
    #         arguments=args,
    #     )

    #     hvals = hcons['h']
    #     mask = numpy.isfinite(hvals)

    #     if key == 'V_mesh':
    #         V_mesh[mask] = hvals[mask]

    #     elif key == 'A_mesh':
    #         A_mesh[mask] = hvals[mask]

    # check_vm_transfer(
    #     'after interface correction',
    #     V_mesh,
    # )

    # ------------------------------------------------------------------
    # 12. Build new_args
    # ------------------------------------------------------------------
    # new_args = dict(args)   # solid d, d0, u0δt, a0δt2 kept as references, unchanged

    # new_args['dm']     = dm_n_dofs
    # new_args['dm0']    = dm0_new
    # new_args['vm0δt']  = vm0dt_new
    # new_args['am0δt2'] = am0dt2_new

    # # ------------------------------------------------------------------
    # # Enforce the interface Newmark history from the SOLID history.
    # #
    # # Bulk interpolation gives the mesh history in the interior, but at the
    # # interface the mesh must move with the solid by construction:
    # #     v_m,Γ = v_s ,  a_m,Γ = a_s .
    # # Nutils stores these as the scaled increments vm0δt = v·δt and
    # # am0δt2 = a·δt², both in units of the dm dof scale (cylinder_radius).
    # # We solve a small boundary least-squares problem for each and splice
    # # the resulting interface dofs into the bulk-projected arrays.
    # # The previous displacement dm0 is likewise set from the solid's own
    # # previous displacement, re-referenced to the new epoch.
    # # ------------------------------------------------------------------
    # try:
    #     dt_s_ = float(dynamic.timestep / 's')
    #     # Derive the projection field from the ACTUAL dm field by renaming
    #     # its argument.  Building a fresh field on the boundary topology
    #     # would have its own dof numbering, and indexing the volume dm
    #     # arrays with that mask would silently write to the wrong dofs.
    #     ns_hist = Namespace()
    #     ns_hist.h = function.replace_arguments(new_ns.dm, [('dm', 'h')])

    #     d0_expr_solid    = function.replace_arguments(new_ns.d, [('d', 'd0')])
    #     u0dt_expr_solid  = function.replace_arguments(new_ns.d, [('d', 'u0δt')])
    #     a0dt2_expr_solid = function.replace_arguments(new_ns.d, [('d', 'a0δt2')])

    #     hist_targets = {
    #         'dm0': d0_expr_solid - d_s_remesh_field,
    #         'vm0δt':  u0dt_expr_solid,
    #         'am0δt2': a0dt2_expr_solid,
    #     }
    #     for key, target in hist_targets.items():
    #         diff_h = ns_hist.h - target
    #         sqr_h = zipped_traction.integral(
    #             _dot2(diff_h, diff_h) * new_ns.dS
    #         ) / domain.cylinder_radius**3
    #         hcons = System(sqr_h, trial='h').solve_constraints(
    #             droptol=1e-9, arguments=args)
    #         hvals = hcons['h']
    #         mask = numpy.isfinite(hvals)
    #         arr = new_args[key].copy()
    #         arr[mask] = hvals[mask]
    #         new_args[key] = arr
    #         log.info(f'[REMESH] interface {key} set from solid history '
    #                  f'({int(mask.sum())} dof components)')
    # except Exception as ex:
    #     log.info(f'[REMESH] interface Newmark history enforcement skipped: {ex}')

    # # ------------------------------------------------------------------
    # # Recompute v_m^new (and hence u_rel^new) from the FINAL mesh history.
    # #
    # # The interface entries of dm0 / vm0δt / am0δt2 were just overwritten
    # # with the solid history, so the v_m implied by new_args is no longer
    # # the vm_new_dofs_ms computed earlier from the bulk-projected history.
    # # Recomputing here restores u_f = v_m + u_rel exactly.
    # # ------------------------------------------------------------------
    # δaδt2_final = (new_args['dm'] - new_args['dm0']
    #                - new_args['vm0δt'] - .5 * new_args['am0δt2']) / beta
    # uδt_final       = new_args['vm0δt'] + new_args['am0δt2'] + gamma * δaδt2_final
    # vm_new_dofs_ms  = uδt_final * R_scale / dt
    # urel_new_dofs   = uf_proj_dofs * u_scale - vm_new_dofs_ms
    # new_args['u']   = urel_new_dofs / u_scale

    
    # new_args['p']      = p_proj_dofs
    # # Interface Lagrange multiplier: zero initial guess (Newton solves for it).
    # new_args['lam']    = numpy.zeros(function.arguments_for(new_res)['lam'].shape)
    # new_args['u0']   = u0_new
    # new_args['a0δt'] = a0dt_new

    # log.info('Fluid solution transferred to new mesh (physical velocity, full Newmark history).')

    # ------------------------------------------------------------------
    # TRANSFER CHECK: compare old/new velocities at same physical points
    # ------------------------------------------------------------------

    # old_vm_check, old_urel_check = function.eval(
    #     [
    #         old_located.bind(v_field),
    #         old_located.bind(urel_field),
    #     ],
    #     arguments=args,
    # )

    # old_vm_check = numpy.asarray(old_vm_check / 'm/s')
    # old_urel_check = numpy.asarray(old_urel_check / 'm/s')
    # old_uf_check = old_vm_check + old_urel_check


    # new_vm_check, new_urel_check = function.eval(
    #     [
    #         new_dof_sample.bind(new_ns.vm),
    #         new_dof_sample.bind(new_ns.urel),
    #     ],
    #     arguments=new_args,
    # )

    # new_vm_check = numpy.asarray(new_vm_check / 'm/s')
    # new_urel_check = numpy.asarray(new_urel_check / 'm/s')
    # new_uf_check = new_vm_check + new_urel_check


    # def report_transfer_error(name, old, new):

    #     err = new - old

    #     abs_err = numpy.linalg.norm(err, axis=1)
    #     old_mag = numpy.linalg.norm(old, axis=1)

    #     rms_err = numpy.sqrt(numpy.mean(abs_err**2))
    #     rms_old = numpy.sqrt(numpy.mean(old_mag**2))

    #     rel_rms = rms_err / max(rms_old, 1e-30)

    #     log.info(
    #         f'[TRANSFER CHECK] {name}: '
    #         f'max={abs_err.max():.6e} m/s, '
    #         f'RMS={rms_err:.6e} m/s, '
    #         f'relRMS={rel_rms:.6e}'
    #     )


    # report_transfer_error(
    #     'u_phys',
    #     old_uf_check,
    #     new_uf_check,
    # )

    # report_transfer_error(
    #     'v_mesh',
    #     old_vm_check,
    #     new_vm_check,
    # )

    # report_transfer_error(
    #     'u_rel',
    #     old_urel_check,
    #     new_urel_check,
    # )


    # # ALE identity on the new mesh
    # ale_identity = (
    #     new_uf_check
    #     - new_vm_check
    #     - new_urel_check
    # )

    # ale_err = numpy.linalg.norm(
    #     ale_identity,
    #     axis=1,
    # )

    # log.info(
    #     f'[TRANSFER CHECK] ALE identity: '
    #     f'max={ale_err.max():.6e} m/s, '
    #     f'RMS={numpy.sqrt(numpy.mean(ale_err**2)):.6e} m/s'
    # )
    # ------------------------------------------------------------------
    # 12. Build new_args
    # ------------------------------------------------------------------

    new_args = dict(args)
    # Solid d, d0, u0δt, a0δt2 remain unchanged.

    # Fresh mesh displacement
    new_args['dm'] = dm_n_dofs

    # Consistent mesh Newmark history
    new_args['dm0']    = dm0_new
    new_args['vm0δt']  = vm0dt_new
    new_args['am0δt2'] = am0dt2_new

    # Current relative velocity
    new_args['u'] = urel_new_dofs

    # Consistent relative-velocity Newmark history
    new_args['u0']   = u0_new
    new_args['a0δt'] = a0dt_new

    # Pressure
    new_args['p'] = p_proj_dofs

    # Interface Lagrange multiplier
    new_args['lam'] = numpy.zeros(function.arguments_for(new_res)['lam'].shape)

    vm_g, urel_g, uf_g, vs_g = function.eval(
        [zipped_traction.bind(new_ns.vm), zipped_traction.bind(new_ns.urel),
        zipped_traction.bind(new_ns.u), zipped_traction.bind(new_ns.v)],
        arguments=new_args)

    vm_vs = numpy.linalg.norm(numpy.asarray((vm_g - vs_g) / 'm/s'), axis=-1)
    urel_mag = numpy.linalg.norm(numpy.asarray(urel_g / 'm/s'), axis=-1)
    uf_vs = numpy.linalg.norm(numpy.asarray((uf_g - vs_g) / 'm/s'), axis=-1)
    vs_ref = max(numpy.linalg.norm(numpy.asarray(vs_g / 'm/s'), axis=-1).max(), 1e-30)

    log.info(f'[INTERFACE VELOCITY] ||v_m-v_s|| max={vm_vs.max():.6e} m/s, rel={vm_vs.max()/vs_ref:.6e}')
    log.info(f'[INTERFACE VELOCITY] ||u_rel||   max={urel_mag.max():.6e} m/s, rel={urel_mag.max()/vs_ref:.6e}')
    log.info(f'[INTERFACE VELOCITY] ||u_f-v_s|| max={uf_vs.max():.6e} m/s, rel={uf_vs.max()/vs_ref:.6e}')

    log.info(
        'Fluid solution transferred to new mesh; '
        'genuine previous-step Newmark history transferred to fresh mesh.'
    )

    # ------------------------------------------------------------------
    # TRANSFER CHECK: current mesh velocity
    # ------------------------------------------------------------------

    old_vm_check = function.eval(old_located.bind(v_field), arguments=args)
    new_vm_check = function.eval(new_dof_sample.bind(new_ns.vm), arguments=new_args)

    old_vm_check = numpy.asarray(old_vm_check / 'm/s')
    new_vm_check = numpy.asarray(new_vm_check / 'm/s')

    abs_err = numpy.linalg.norm(new_vm_check - old_vm_check, axis=1)
    old_mag = numpy.linalg.norm(old_vm_check, axis=1)

    rms_err = numpy.sqrt(numpy.mean(abs_err**2))
    rms_old = numpy.sqrt(numpy.mean(old_mag**2))
    rel_rms = rms_err / max(rms_old, 1e-30)

    log.info(f'[TRANSFER CHECK] v_mesh: max={abs_err.max():.6e} m/s, RMS={rms_err:.6e} m/s, relRMS={rel_rms:.6e}')
    

    # ------------------------------------------------------------------
    # 13. Verification checks
    # ------------------------------------------------------------------
    from scipy.spatial import cKDTree

    new_fluid_bbezier_check = new_topo['fluid'].boundary['cylinder'].sample('bezier', 3)
    xb_fluid_inner = numpy.asarray(new_fluid_bbezier_check.eval(new_geom) / 'm', dtype=float)
    xb_solid_outer = numpy.asarray(xb_current_m)

    tree = cKDTree(xb_fluid_inner)
    dists, _ = tree.query(xb_solid_outer)
    log.info(f'[REMESH CHECK] Interface gap max : {dists.max():.3e} m')
    log.info(f'[REMESH CHECK] Interface gap mean: {dists.mean():.3e} m')

    dm_max = numpy.abs(new_args['dm']).max()
    log.info(f'[REMESH CHECK] max |dm| after reset: {dm_max:.3e}  (expect ~0)')
    log.info(f'[REMESH CHECK] max |v_m^new| : {numpy.abs(vm_new_dofs_ms).max():.3e} m/s  (need not be 0)')

    # Kinematic interface consistency: the mesh must move with the solid,
    # i.e. v_m = v_s and a_m = a_s on the cylinder.  The bulk interpolation
    # used for the dm Newmark history does NOT guarantee this at the new
    # interface dofs, so we measure it explicitly.  Large values here mean
    # the interface histories should be taken from the solid rather than
    # from bulk interpolation.
    try:
        vm_gamma, am_gamma, vs_gamma, as_gamma = function.eval(
            [zipped_traction.bind(new_ns.vm), zipped_traction.bind(new_ns.am),
             zipped_traction.bind(new_ns.v),  zipped_traction.bind(new_ns.a)],
            arguments=new_args,
        )
        dv = numpy.linalg.norm(numpy.asarray((vm_gamma - vs_gamma) / 'm/s'), axis=-1)
        da = numpy.linalg.norm(numpy.asarray((am_gamma - as_gamma) / 'm/s2'), axis=-1)
        vs_ref = max(numpy.linalg.norm(numpy.asarray(vs_gamma / 'm/s'), axis=-1).max(), 1e-30)
        as_ref = max(numpy.linalg.norm(numpy.asarray(as_gamma / 'm/s2'), axis=-1).max(), 1e-30)
        log.info(f'[REMESH CHECK] ||v_m - v_s||_Γ max: {dv.max():.3e} m/s   '
                 f'(rel {dv.max()/vs_ref:.3e})')
        log.info(f'[REMESH CHECK] ||a_m - a_s||_Γ max: {da.max():.3e} m/s2  '
                 f'(rel {da.max()/as_ref:.3e})')
        if dv.max() > 1e-3 * vs_ref:
            log.info('[REMESH WARNING] interface mesh/solid velocity mismatch exceeds diagnostic tolerance.')
        interface_dv = float(dv.max()); interface_da = float(da.max())
    except Exception as ex:
        log.info(f'[REMESH CHECK] interface kinematics check skipped: {ex}')
        interface_dv = interface_da = float('nan')

    log.info('=== REMESHING COMPLETE ===')

    # ------------------------------------------------------------------
    # 14. Rebuild Bezier samplers and diagnostic quantities
    # ------------------------------------------------------------------
    new_fluid_bezier = new_topo['fluid'].sample('bezier', 3)

    new_x_bz  = function.factor(new_fluid_bezier.bind(ns_f.xm))
    new_u_bz  = function.factor(new_fluid_bezier.bind(new_ns.u))
    new_p_bz  = function.factor(new_fluid_bezier.bind(new_ns.p))
    new_Jfluid_bz = new_fluid_bezier.bind(ns_m.Jm)

    new_stagnation_sample = new_topo['fluid'].sample('gauss', 4)
    new_x_stag    = new_stagnation_sample.bind(ns_f.xm)
    new_u_stag    = new_stagnation_sample.bind(new_ns.u)
    new_edot_stag = new_stagnation_sample.bind(ns_f.edot)

    # ------------------------------------------------------------------
    # 15. Updated fluid_state for the NEXT remesh (or for per-step use)
    # ------------------------------------------------------------------
    new_fluid_state = dict(
        topo_fluid = new_topo['fluid'],
        geom_bare  = new_geom_bare,

        x_field    = ns_f.xm,

        v_field    = new_ns.vm,
        a_field    = new_ns.am,

        urel_field = new_ns.urel,
        arel_field = new_ns.arel,

        p_field    = new_ns.p,

        d_field    = new_ns.dm,
        d_name     = 'dm',
        d0_key     = 'dm0',
        u0dt_key   = 'vm0δt',
        a0dt2_key  = 'am0δt2',

        # Ingredients for the per-timestep interface update
        dm_zipped_interface = zipped_traction,
        d_s_remesh_const_m  = d_s_remesh_const_m,

        # Fixed outer-boundary constraint and interface data
        dm_outer_cons       = dm_outer_cons,
        solid_cyl_gauss     = solid_cyl_gauss,
    )

    return (
        new_ns, new_res, new_cons, new_ucons, new_system,
        new_fluid_bezier,
        new_x_bz, new_u_bz, new_p_bz,
        new_stagnation_sample, new_x_stag, new_u_stag, new_edot_stag,
        new_hydrodynamic_force,
        new_Jfluid_bz,
        new_fluid_state,
        new_args,
        dict(
            interface_gap_max  = float(dists.max()),
            interface_gap_mean = float(dists.mean()),
            dm_max_after_reset = float(dm_max),
            vm_new_max         = float(numpy.abs(vm_new_dofs_ms).max()),
            interface_dv       = interface_dv,
            interface_da       = interface_da,
        ),
    )

def remesh_diagnostics(
    current_t_s,
    remesh_count,
    Jf_old,
    Jf_new,
    old_fluid_bezier,
    new_fluid_bezier,
    xb_solid_m,
    old_x_m,
    new_x_m,
    old_u_vals,
    new_u_vals,
    old_p_vals,
    new_p_vals,
    domain,
    transfer_metrics,
):
    """Produce all diagnostic plots and prints at a remesh event.

    Called immediately after ``remesh_fluid`` returns.  Generates:
    - console summary of mesh quality before/after
    - side-by-side mesh comparison plot
    - side-by-side velocity-magnitude comparison plot
    - zoomed interface-alignment plot
    - transfer error metrics in the log
    """

    from scipy.spatial import cKDTree

    L_plot = float(domain.channel_length / 'm')
    R_plot = float(domain.cylinder_radius / 'm')
    cx = xb_solid_m[:, 0].mean()
    cy = xb_solid_m[:, 1].mean()

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    log.info(f'--- Remesh #{remesh_count} at t={current_t_s:.6e} s ---')
    log.info(f'  Jf_min  before : {Jf_old.min():.4f}')
    log.info(f'  Jf_min  after  : {Jf_new.min():.4f}')
    log.info(f'  Jf_max  before : {Jf_old.max():.4f}')
    log.info(f'  Jf_max  after  : {Jf_new.max():.4f}')
    log.info(f'  Interface gap max  : {transfer_metrics["interface_gap_max"]:.3e} m')
    log.info(f'  Interface gap mean : {transfer_metrics["interface_gap_mean"]:.3e} m')
    log.info(f'  dm max after reset : {transfer_metrics["dm_max_after_reset"]:.3e}  (expect ~0)')
    log.info(f'  v_m max after transfer : {transfer_metrics["vm_new_max"]:.3e} m/s  (need not be 0)')

    # ------------------------------------------------------------------
    # Plot 1: mesh comparison (old deformed ALE vs new clean mesh)
    # ------------------------------------------------------------------
    with export.mplfigure(f'remesh_{remesh_count:03d}_mesh_t{current_t_s:.6f}s.jpg', dpi=200) as fig:

        ax1 = fig.add_subplot(121, title=f'ALE mesh before remesh\nJf_min={Jf_old.min():.3f}')
        export.triplot(ax1, old_x_m, hull=old_fluid_bezier.hull, linewidth=0.25)
        closed = numpy.vstack([xb_solid_m, xb_solid_m[:1]])
        ax1.plot(closed[:, 0], closed[:, 1], 'r-', linewidth=1.5, label='solid boundary')
        ax1.set_aspect('equal')
        ax1.set_xlim(-L_plot, L_plot)
        ax1.set_ylim(-L_plot, L_plot)
        ax1.legend(fontsize=6)

        ax2 = fig.add_subplot(122, title=f'New fluid mesh after remesh\nJf_min={Jf_new.min():.3f}')
        export.triplot(ax2, new_x_m, hull=new_fluid_bezier.hull, linewidth=0.25)
        ax2.plot(closed[:, 0], closed[:, 1], 'r-', linewidth=1.5, label='solid boundary')
        ax2.set_aspect('equal')
        ax2.set_xlim(-L_plot, L_plot)
        ax2.set_ylim(-L_plot, L_plot)
        ax2.legend(fontsize=6)

        fig.suptitle(f'Remesh #{remesh_count}  t={current_t_s:.4e} s', fontsize=9)

    # ------------------------------------------------------------------
    # Plot 2: velocity magnitude comparison (old vs new mesh)
    # ------------------------------------------------------------------
    vmin = min(old_u_vals.min(), new_u_vals.min())
    vmax = max(old_u_vals.max(), new_u_vals.max())

    with export.mplfigure(f'remesh_{remesh_count:03d}_velocity_t{current_t_s:.6f}s.jpg', dpi=200) as fig:

        ax1 = fig.add_subplot(121, title='|u| before remesh  [m/s]')
        im = ax1.tripcolor(
            old_x_m[:, 0], old_x_m[:, 1], old_fluid_bezier.tri,
            old_u_vals, shading='gouraud', vmin=vmin, vmax=vmax,
        )
        ax1.plot(closed[:, 0], closed[:, 1], 'w-', linewidth=1)
        ax1.set_aspect('equal')
        ax1.set_xlim(-L_plot, L_plot)
        ax1.set_ylim(-L_plot, L_plot)

        ax2 = fig.add_subplot(122, title='|u| after remesh  [m/s]')
        im2 = ax2.tripcolor(
            new_x_m[:, 0], new_x_m[:, 1], new_fluid_bezier.tri,
            new_u_vals, shading='gouraud', vmin=vmin, vmax=vmax,
        )
        ax2.plot(closed[:, 0], closed[:, 1], 'w-', linewidth=1)
        ax2.set_aspect('equal')
        ax2.set_xlim(-L_plot, L_plot)
        ax2.set_ylim(-L_plot, L_plot)

        fig.colorbar(im2, ax=[ax1, ax2], orientation='horizontal', label='|u| [m/s]', shrink=0.6)
        fig.suptitle(f'Remesh #{remesh_count}  t={current_t_s:.4e} s', fontsize=9)

    # ------------------------------------------------------------------
    # Plot 3: zoomed interface alignment
    # ------------------------------------------------------------------
    zoom = 3.0 * R_plot

    with export.mplfigure(f'remesh_{remesh_count:03d}_interface_t{current_t_s:.6f}s.jpg', dpi=250) as fig:

        ax = fig.add_subplot(111, title='Interface alignment (solid = red, new fluid inner = blue)')
        export.triplot(ax, new_x_m, hull=new_fluid_bezier.hull, linewidth=0.3)
        ax.plot(closed[:, 0], closed[:, 1], 'r-', linewidth=2.5, label='solid boundary', zorder=5)

        # New fluid inner boundary — use the ordered solid boundary points
        # (already confirmed to coincide with new fluid boundary at machine precision)
        ax.scatter(xb_solid_m[:, 0], xb_solid_m[:, 1],
                   s=8, c='blue', zorder=6, label='new fluid boundary nodes')

        ax.set_aspect('equal')
        ax.set_xlim(cx - zoom, cx + zoom)
        ax.set_ylim(cy - zoom, cy + zoom)
        ax.legend(fontsize=7)
        ax.set_title(
            f'Interface alignment — remesh #{remesh_count}\n'
            f'gap_max={transfer_metrics["interface_gap_max"]:.2e} m',
            fontsize=8,
        )


@dataclass
class Domain:
    '''Parameters for the domain geometry.

    The default values match Table 1 of Turek and Hron [1].'''

    channel_length: Length = Length('0.0008m')
    channel_height: Length = Length('0.0004m')
    cylinder_radius: Length = Length('0.00003m')
    
    elemsize: Length = Length('0.00002m')
    coarsening: float = 4.

    def generate_mesh(self):
        'Call gmsh to generate mesh and return topo, geom tuple.'''

        u = Length('m') # reference length for mesh generation

        topo, geom = gmsh(
            Path(__file__).parent/'crosschannel_2.geo',
            dimension=2,
            order=2,
            numbers={
                 'channel_length': self.channel_length/u,
                 'channel_height': self.channel_height/u,
                 'cylinder_radius': self.cylinder_radius/u,
                 'elemsize': self.elemsize/u,
                 'coarsening': self.coarsening})

        bezier = topo.sample('bezier', 2)
        bezier_cylinder = topo['fluid'].boundary['cylinder'].sample('bezier', 3)
        #A = topo.points['A'].sample('gauss', 1).eval(geom)

        with export.mplfigure('mesh.jpg', dpi=150) as fig:
            ax = fig.add_subplot(111)
            export.triplot(ax, bezier.eval(geom), hull=bezier.hull)
            export.triplot(ax, bezier_cylinder.eval(geom), hull=bezier_cylinder.tri, linewidth=1, linecolor='b')

            L = self.channel_length/u
            ax.set_xlim(-L, L)
            ax.set_ylim(-L, L)
            
 
        return topo, geom * u


@dataclass
class Solid:
    '''Parameters for the solid problem.'''

    density: Density = Density('10kg/L')
    poisson_ratio: float = .49
    shear_modulus: Pressure = Pressure('0.0069444MPa')
    gravity: Acceleration = Acceleration('0m/s2')

    def lame_parameters(self):
        'Return tuple of first and second lame parameter.'

        return 2 * self.shear_modulus * self.poisson_ratio / (1 - 2 * self.poisson_ratio), self.shear_modulus

    def young(self):
        "Return Young's elasticity modulus."

        return 2 * self.shear_modulus * (1 + self.poisson_ratio)


@dataclass
class Fluid:
    '''Parameters for the fluid problem.'''

    density: Density = Density('10kg/L')
    viscosity: Viscosity = Viscosity('0.00559Pa*s')
    velocity: Velocity = Velocity('0.745m/s')

    def reynolds(self, reference_length):
        'Return Reynolds number for given reference length'

        return self.density * self.velocity * reference_length / self.viscosity


@dataclass
class Dynamic:
    '''Parameters relating to time dependence.'''

    timestep: Time = Time('0.01ms') # simulation time step size
    endtime: Time = Time('0.5s') # total duration of the simulation
    init: Time = Time('.5s') # duration of the ramp-up phase
    window: Time = Time('1s') # sliding window length for time series plots
    gamma: float = .55
    beta: float = .275625

    def __post_init__(self):
        self.timeseries = defaultdict(deque(maxlen=round(self.window / self.timestep)).copy)

    def ramp_up(self, t):
        'Return inflow ramp-up scale factor at given time.'

        return .5 - .5 * numpy.cos(numpy.pi * min(t / self.init, 1))

    @property
    def times(self):
        'Return all configured time steps for the simulation.'

        return numpy.arange(1, self.endtime / self.timestep + .5) * self.timestep

    def add_and_plot(self, name, t, v, ax):
        'Add data point and plot time series for past window.'

        d = self.timeseries[name]
        d.append((t, v))
        times, values = numpy.stack(d, axis=1)
        ax.plot(times, values)
        ax.set_ylabel(name)
        ax.grid()
        ax.autoscale(enable=True, axis='x', tight=True)
        vmin, vmax = numpy.quantile(values, [0,1])
        vmean = (vmax + vmin) / 2
        values -= vmean
        icross, = numpy.nonzero(values[1:] * values[:-1] < 0)
        if len(icross) >= 4: # minimum of two up, two down
            tcross = (times[icross] * values[icross+1] - times[icross+1] * values[icross]) / (values[icross+1] - values[icross])
            ax.plot(tcross, [vmean] * len(icross), '+')
            ax.text(tcross[numpy.diff(tcross).argmax():][:2].mean(), vmean,
                s=f'{vmean:+.4f}\n±{(vmax-vmin)/2:.4f}\n↻{(tcross[2:]-tcross[:-2]).mean():.4f}',
                va='center', ha='center', multialignment='right')

    # The Newmark-beta scheme is used for time integration. For our formulation
    # to support both stationary and dynamic simulations, we take displacement
    # (solid) and velocity (fluid) as primary variables, with time derivatives
    # introduced via helper arguments that are updated after every solve.
    #
    # d = d0 + δt u0 + .5 δt^2 aβ, where aβ = (1-2β) a0 + 2β a
    # => δd = δt u0 + δt^2 [ .5 a0 + β δa ]
    # => δa = [ δd / δt^2 - u0 / δt - .5 a0 ] / β
    #
    # u = u0 + δt aγ, where aγ = (1-γ) a0 + γ a
    # => δu = δt [ a0 + γ δa ]
    # => δa = [ δu / δt - a0 ] / γ

    def newmark_defo_args(self, d, d0=0., u0δt=0., a0δt2=0., **args):
        δaδt2 = (d - d0 - u0δt - .5 * a0δt2) / self.beta
        uδt = u0δt + a0δt2 + self.gamma * δaδt2
        aδt2 = a0δt2 + δaδt2
        return dict(args, d=d+uδt+.5*aδt2, d0=d, u0δt=uδt, a0δt2=aδt2)

    def newmark_defo(self, d):
        D = self.newmark_defo_args(d, *[function.replace_arguments(d, [('d', t)]) for t in ('d0', 'u0δt', 'a0δt2')])
        return D['u0δt'] / self.timestep, D['a0δt2'] / self.timestep**2

    def newmark_velo_args(self, u, u0=0., a0δt=0., **args):
        aδt = a0δt + (u - u0 - a0δt) / self.gamma
        return dict(args, u=u+aδt, u0=u, a0δt=aδt)

    def newmark_velo(self, u):
        D = self.newmark_velo_args(u, *[function.replace_arguments(u, [('u', t)]) for t in ('u0', 'a0δt')])
        return D['a0δt'] / self.timestep

    # ------------------------------------------------------------------
    # Generic (named-argument) Newmark helpers.
    #
    # The methods above hardcode the argument names 'd', 'd0', 'u0δt',
    # 'a0δt2' (displacement-type) and 'u', 'u0', 'a0δt' (velocity-type).
    # After remeshing, the fluid mesh displacement field is named 'dm'
    # (to keep it distinct from the persistent solid field 'd'), so we
    # need equivalents that work with arbitrary argument names.
    # ------------------------------------------------------------------

    def newmark_defo_args_named(self, args, name, d0_name, u0dt_name, a0dt2_name):
        'Numeric Newmark predictor step for an arbitrarily-named displacement field.'

        d     = args[name]
        d0    = args.get(d0_name, 0.)
        u0dt  = args.get(u0dt_name, 0.)
        a0dt2 = args.get(a0dt2_name, 0.)
        δaδt2 = (d - d0 - u0dt - .5 * a0dt2) / self.beta
        uδt   = u0dt + a0dt2 + self.gamma * δaδt2
        aδt2  = a0dt2 + δaδt2
        new_args = dict(args)
        new_args[name]       = d + uδt + .5 * aδt2
        new_args[d0_name]    = d
        new_args[u0dt_name]  = uδt
        new_args[a0dt2_name] = aδt2
        return new_args

    def newmark_defo_named(self, d, name, d0_name, u0dt_name, a0dt2_name):
        'Symbolic Newmark velocity/acceleration for an arbitrarily-named displacement field.'

        d0    = function.replace_arguments(d, [(name, d0_name)])
        u0dt  = function.replace_arguments(d, [(name, u0dt_name)])
        a0dt2 = function.replace_arguments(d, [(name, a0dt2_name)])
        δaδt2 = (d - d0 - u0dt - .5 * a0dt2) / self.beta
        uδt   = u0dt + a0dt2 + self.gamma * δaδt2
        aδt2  = a0dt2 + δaδt2
        return uδt / self.timestep, aδt2 / self.timestep**2


def main(domain: Domain = Domain(), solid: Optional[Solid] = Solid(), fluid: Optional[Fluid] = Fluid(), dynamic: Optional[Dynamic] = Dynamic(), residence_radius_factor: float = 1.0, stop_y_max: Length = Length('0.0006m')):
    '''Turek Hron benchmark problem

    This is a monolythic ALE (Arbitrary Lagrangian Eulerian) implementation of
    the fluid-structure interaction benchmark defined in 2006 by [Turek and
    Hron](https://doi.org/10.1007/3-540-34596-5_15). The implementation covers
    all fluid dynamics tests CFD1, CFD2 and CFD3, all structural tests CSM1,
    CSM2 and CSM3, and all interaction tests FSI1, FSI2 and FSI3, as well as
    freely defined variants thereof.'''

    assert solid or fluid, 'nothing to compute'

    if fluid:
        log.info('Re:', fluid.reynolds(domain.cylinder_radius))
        if solid:
            log.info('Ca_e:', fluid.velocity * fluid.viscosity / (solid.shear_modulus *  domain.cylinder_radius))
            #log.info('Ae:', solid.young() / fluid.density / fluid.velocity**2)
            log.info('β:',  solid.density / fluid.density)

    topo, geom = domain.generate_mesh()

    fluid_bezier = topo['fluid'].sample('bezier', 3)              
    bbezier      = topo['solid'].boundary['cylinder'].sample('bezier', 3)
    solid_bezier = topo['solid'].sample('bezier', 3)            


    res = 0.
    cons = {}
    args = {}

    ns = Namespace()
    ns.δ = function.eye(2)
    ns.xref = geom
    ns.define_for('xref', gradient='∇ref', jacobians=('dVref', 'dSref'))

    if solid:

        ns.ρs = solid.density
        ns.λs, ns.μs = solid.lame_parameters()
        ns.g = -solid.gravity * ns.δ[1]

        # Deformation, velocity and acceleration are defined on the entire
        # domain, solving for conservation of momentum on the solid domain and
        # a mesh continuation problem on the fluid domain. While the latter is
        # used only when fluid is enabled, we include it regardless both for
        # simplicity and for testing purposes.
        ns.d = topo.field('d', btype='std', degree=2, shape=(2,)) * domain.cylinder_radius # deformation at the end of the timestep
        if dynamic:
            ns.v, ns.a = dynamic.newmark_defo(ns.d)
        else:
            ns.a = Acceleration.wrap(function.zeros((2,)))

        # Deformed geometry
        ns.δ_ij = 'δ_ij'

        ns.x_i = 'xref_i + d_i'
        ns.F_ij = '∇ref_j(x_i)' # deformation gradient tensor
        ns.J = numpy.linalg.det(ns.F)
        ns.C_ij = 'F_ki F_kj' # right Cauchy-Green deformation tensor
        ns.Cinv = numpy.linalg.inv(ns.C)
        ns.E_ij = '.5 (C_ij - δ_ij)' # Green-Lagrangian strain tensor
        ns.S_ij     = 'μs (δ_ij - Cinv_ij) + λs ln(J) Cinv_ij' # 2nd Piola–Kirchhoff stress
        ns.P_ij     = 'F_ik S_kj'
        ns.σcauchy_ij = 'P_ik F_jk / J'
        ns.Finv         = numpy.linalg.inv(ns.F)
        
        ns.MeshPK_ij      = '2 (F_ij - Finv_ji)'
        # Solid deformation measures
        ns.Eeq2 = '2 E_ij E_ij'
        ns.Psi = '.5 μs (C_kk - 2 - 2 ln(J)) + .5 λs ln(J)^2' #'.5 λs E_kk E_ll + μs E_ij E_ij'


        # Momentum balance: ρs (a - g) = div P
        ns.dtest = function.replace_arguments(ns.d, 'd:dtest') / (solid.shear_modulus * domain.cylinder_radius**2)
        res += topo['solid'].integral('(∇ref_j(dtest_i) P_ij + dtest_i ρs (a_i - g_i)) dVref' @ ns, degree=4)

        # In the momentum balance above, the only test and trial dofs involved
        # are those that have support on the solid domain. The remaining trial
        # dofs will follow from minimizing a mesh energy functional, using the
        # solid deformation as a boundary constraint so that the continuation
        # problem does not feed back into the physics. To achieve this within a
        # monolythic setting, we establish a boolean array 'dfluid' to select
        # all dofs that are exclusively supported by the fluid domain, and use
        # it to restrict the minimization problem to the remaining dofs via the
        # linearize operation.
        mesh_energy = topo['fluid'].integral('C_kk - 2 log(J)' @ ns, degree=4) # Neo-Hookean with jacobian based stiffness
        #mesh_energy = topo['fluid'].integral('∇ref_k(d_i) ∇ref_k(d_i) dVref' @ ns,degree=4)/ (domain.cylinder_radius**2)
        sqr = topo['solid'].integral('d_k d_k dVref' @ ns, degree=4) / domain.cylinder_radius**4
        dfluid = numpy.isnan(System(sqr, trial='d').solve_constraints(droptol=1e-9)['d']) # true if dof is not supported by solid domain

        dfluid_mask = dfluid  # True  = fluid-bulk d DOFs  → mesh extension
        if fluid:
            res += Pressure('1Pa') * function.replace_arguments(topo['fluid'].integral('∇ref_j(dtest_i) MeshPK_ij dVref' @ ns, degree=4,
             ),
             {'dtest': function.arguments_for(res)['dtest'] * dfluid_mask},
             )
        #res += function.linearize(mesh_energy, {'d': function.arguments_for(res)['dtest'] * dfluid})

        # Deformation constraints: fixed at exterior boundary and cylinder
        sqr = topo.boundary.integral('d_k d_k dSref' @ ns, degree=4) / domain.cylinder_radius**3
        cons = System(sqr, trial='d').solve_constraints(droptol=1e-9, constrain=cons)

        # Zero initial deformation
        args['d'] = numpy.zeros(function.arguments_for(res)['d'].shape)

    else: # fully rigid solid

        ns.x = ns.xref
        ns.v = Velocity.wrap(function.zeros((2,)))
        ns.a = Acceleration.wrap(function.zeros((2,)))

    if fluid:

        ns.ρf = fluid.density
        ns.μf = fluid.viscosity

        ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))

        # The fluid velocity is defined relative to the mesh velocity, which
        # helps to impose the no slip condition on the elastic structure.
        # Furthermore, in ALE the material derivative convects only with this
        # relative component.
        ns.urel = topo['fluid'].field('u', btype='std', degree=2, shape=(2,)) * fluid.velocity
        if dynamic:
            ns.arel = dynamic.newmark_velo(ns.urel)
            ns.u_i = 'v_i + urel_i'
            ns.DuDt_i = 'a_i + arel_i + ∇_j(u_i) urel_j' # material derivative
        else:
            ns.u = ns.urel
            ns.DuDt_i = '∇_j(u_i) u_j'
        # rate of deformation tensor 
        ns.edot_ij = '.5 (∇_j(u_i) + ∇_i(u_j))'
        
        #sample points for stagnation point detection
        stagnation_sample = topo['fluid'].sample('gauss', 4)
        x_stag_sample = stagnation_sample.bind(ns.x)
        u_stag_sample = stagnation_sample.bind(ns.u)
        edot_stag_sample = stagnation_sample.bind(ns.edot)

        ns.p = topo['fluid'].field('p', btype='std', degree=1) * fluid.viscosity * fluid.velocity / domain.cylinder_radius
        ns.σ_ij = 'μf (∇_j(u_i) + ∇_i(u_j)) - p δ_ij' # fluid stress tensor
        hydrodynamic_force = topo['fluid'].boundary['cylinder'].integral('-σ_ij n_j dS' @ ns, degree=4)

        # Project Posseuille inflow at inlet (exact because of quadratic
        # velocity basis), parallel outflow at outlet, and no-slip conditions
        # for the relative velocity at the remaining boundaries.
        y = ns.xref[1] / domain.channel_height
        H = domain.channel_height/ domain.channel_height        

        ns.uin = fluid.velocity * (1 - (2*y/H)**2)
        sqr = topo['fluid'].boundary['wall,cylinder'].integral('urel_k urel_k dSref' @ ns, degree=4) / (domain.cylinder_radius * fluid.velocity**2)
        sqr += (topo['fluid'].boundary['inlet1'].integral('(urel_0 - uin)^2 dSref' @ ns, degree=4)/ (domain.cylinder_radius * fluid.velocity**2))
        sqr += (topo['fluid'].boundary['inlet2'].integral('(urel_0 + uin)^2 dSref' @ ns, degree=4)/ (domain.cylinder_radius * fluid.velocity**2))
        sqr += (topo['fluid'].boundary['inlet1,inlet2'].integral('urel_1^2 dSref' @ ns, degree=4)/ (domain.cylinder_radius * fluid.velocity**2))
        #sqr += (topo['fluid'].boundary['outlet1,outlet2'].integral('urel_0^2 dSref' @ ns, degree=4)/ (domain.cylinder_radius * fluid.velocity**2))
        cons = System(sqr, trial='u').solve_constraints(droptol=1e-9, constrain=cons) # exact projection
        ucons = cons['u'] # save for ramp-up phase

        # Momentum balance: ρf Du/Dt = ∇σ => ∀v ∫ (v ρf Du/Dt + ∇v:σ) = ∮ v·σ·n = 0
        ns.utest = function.replace_arguments(ns.urel, 'u:utest') / fluid.viscosity / fluid.velocity**2
        res += topo['fluid'].integral('(utest_i ρf DuDt_i + ∇_j(utest_i) σ_ij) dV' @ ns, degree=4)

        # Incompressibility: div u = 0
        ns.ptest = function.replace_arguments(ns.p, 'p:ptest') / fluid.viscosity / fluid.velocity**2
        res += topo['fluid'].integral('ptest ∇_k(u_k) dV' @ ns, degree=4)

        if solid:
            # The action of fluid stress on the solid is imposed weakly by
            # lifting the test functions into the fluid domain, using the
            # identity ∮ d·σ·n = ∫ ∇(d·σ) = ∫ (∇d:σ + d·∇σ) = ∫ (∇d:σ + ρf
            # d·Du/Dt). We need the inverse of the dfluid mask to exclude dofs
            # without support on the boundary.
            dsolid = ~dfluid # true if dof is (partially) supported by solid domain
            res += function.replace_arguments(
                topo['fluid'].integral('(dtest_i ρf DuDt_i + ∇_j(dtest_i) σ_ij) dV' @ ns, degree=4),
                {'dtest': function.arguments_for(res)['dtest'] * dsolid})

        # The net force will likewise be evaluated weakly, via the identity F =
        # ∮ σ·n = ∮ λ σ·n = ∫ ∇(λ σ) = ∫ (∇λ·σ + λ ∇σ) = ∫ (∇λ·σ + λ ρf Du/Dt),
        # with λ = 1 on the boundary and lifted into the fluid domain.
        lift = topo['fluid'].field('lift', btype='std', degree=2) # same basis as urel, but scalar
        #sqr = topo['fluid'].boundary['cylinder'].integral((lift - 1)**2, degree=4)
        #lcons = System(sqr, trial='lift').solve_constraints(droptol=1e-9)
        #ns.λ = function.replace_arguments(lift, {'lift': numpy.nan_to_num(lcons['lift'])})
        #F = topo['fluid'].integral('-(∇_j(λ) σ_ij + λ ρf DuDt_i) dV' @ ns, degree=4)

        # Zero initial velocity
        args['u'] = numpy.zeros(function.arguments_for(res)['u'].shape)

        u_bz = function.factor(fluid_bezier.bind(ns.u))
        p_bz = function.factor(fluid_bezier.bind(ns.p)) 


    x_bz = function.factor(fluid_bezier.bind(ns.x))
    x_bbz = function.factor(bbezier.bind(ns.x))

    Jfluid_bz = fluid_bezier.bind(ns.J)

    # Generic "current fluid state" tracking, used by remesh_fluid so that
    # every remesh (first, second, ...) samples the CURRENT fluid fields
    # correctly, whatever they happen to be named after previous remeshes.
    fluid_state = dict(
        topo_fluid = topo['fluid'],
        geom_bare  = geom / Length('1m'),

        x_field    = ns.x,

        v_field    = ns.v,
        a_field    = ns.a,

        urel_field = ns.urel,
        arel_field = ns.arel,

        p_field    = ns.p,

        d_field    = ns.d,
        d_name     = 'd',
        d0_key     = 'd0',
        u0dt_key   = 'u0δt',
        a0dt2_key  = 'a0δt2',

        dm_zipped_interface = None,
        d_s_remesh_const_m  = None,
    )
    if solid:
        solid_bz = topo['solid'].sample('bezier',3)
        xsolid_bz = function.factor(solid_bezier.bind(ns.x))
        Jsolid_bz = solid_bz.bind(ns.J)
        d_cyl_bz = function.factor(bbezier.bind(ns.d))
        
        # ------------------------------------------------------------
        # Particle material center of mass and geometric centroid
        # ------------------------------------------------------------

        # Material center of mass:
        # x_cm = ∫ ρs x dVref / ∫ ρs dVref
        solid_mass_ref = topo['solid'].integral('ρs dVref' @ ns, degree=4)
        solid_moment_ref = topo['solid'].integral('ρs x_i dVref' @ ns, degree=4)
        solid_center_material = solid_moment_ref / solid_mass_ref

        solid_velocity_material = topo['solid'].integral('ρs v_i dVref' @ ns, degree=4) / solid_mass_ref
        

        # Geometric centroid of the current/deformed particle shape:
        # x_geo = ∫ x dV / ∫ dV = ∫ x J dVref / ∫ J dVref
        solid_volume_current = topo['solid'].integral('J dVref' @ ns, degree=4)
        solid_moment_current = topo['solid'].integral('x_i J dVref' @ ns, degree=4)
        solid_center_geometric = solid_moment_current / solid_volume_current

        # CSV file
        centroid_csv = Path('Re80_no_remesh_bg.csv')
        centroid_csv.write_text(
            'time,step,'
            'xp,yp,'
            'xgeo,ygeo,'
            'vxp,vyp,'
            'Fx,Fy,'
            'x_stag,y_stag,'
            'minimum_speed,'
            'epsdot_sp,'
            'particle_max_abs_y,'
            'L_particle,B_particle,'
            'a_x_m,a_y_m,'
            'D_xy,'
            'deformation_index,orientation_rad,'
            'residence_flag,residence_time,'
            'distance_from_center_m,'
            'Jf_min,Jf_max,'
            'remesh_flag,'
            'probe0_umag,probe1_umag,probe2_umag,'
            'probe0_p,probe1_p,probe2_p\n'
        )

        # Fixed spatial probe points for continuity monitoring across remesh events.
        # Chosen to be away from the particle and in different flow regions.
        probe_points_m = numpy.array([
            [-3e-4,  1e-4],   # upstream-left
            [ 3e-4,  1e-4],   # downstream-right
            [ 0.0,   1.5e-4], # top, on axis
        ])

        # Residence-time definition:
        # the particle is counted as resident when its geometric centroid is
        # within residence_radius_factor * particle_radius of the cross-slot
        # center/stagnation region.
        residence_radius_m = residence_radius_factor * float(domain.cylinder_radius / 'm')
        residence_time_s = 0.0

    trial = 'upd'[0 if fluid else 2:2 if not solid else 3]
    system = System(res, trial=list(trial), test=[t+'test' for t in trial])

    DL = uxy = None # for unit tests only
    previous_t_s = 0.0
    remesh_count = 0

    # ------------------------------------------------------------------
    # Remeshing control
    # ------------------------------------------------------------------
    # Trigger remeshing when the minimum Jacobian of the fluid mesh drops
    # below this threshold (relative to a perfect element = 1).
    enable_remeshing = False
    remesh_J_threshold = 0.3

    # Also allow a time-interval-based fallback trigger (set to None to disable).
    # Set to a small value for an early forced remesh to test the procedure
    # before the mesh actually degrades. Once validated, set back to None.
    remesh_interval_s  = None  # first remesh forced at t=0.0001 s for testing

    next_remesh_time_s = remesh_interval_s if remesh_interval_s else float('inf')

    if solid:
        solid_area_ref = topo['solid'].integral('dVref' @ ns, degree=4)
        Eeq2_mean = topo['solid'].integral('Eeq2 dVref' @ ns, degree=4) / solid_area_ref
        Psi_mean = topo['solid'].integral('Psi dVref' @ ns, degree=4) / solid_area_ref
        Psi_over_G_mean = Psi_mean / solid.shear_modulus

    # Track whether we have remeshed (changes which Newmark keys exist)
    has_remeshed = False

    for istep, t in enumerate(log.iter.fraction('timestep', dynamic.times) if dynamic else [Time.wrap(float('inf'))], start=1):

    # ------------------------------------------------------------
    # Time integration predictor/update
    # ------------------------------------------------------------
        if dynamic:
            if solid:
                args = dynamic.newmark_defo_args(**args)

            if fluid:
                args = dynamic.newmark_velo_args(**args)
                cons['u'] = ucons * dynamic.ramp_up(t)

            # After remesh, also advance d_m via the generic named Newmark
            # helper (dm/dm0/vm0δt/am0δt2 — NOT the hardcoded d/d0/u0δt/a0δt2
            # keys, which belong to the persistent solid field).
            if has_remeshed and 'dm' in args:
                args = dynamic.newmark_defo_args_named(
                    args, name='dm', d0_name='dm0',
                    u0dt_name='vm0δt', a0dt2_name='am0δt2',
                )

        # ------------------------------------------------------------
        # Refresh the P_Γ interface constraint for d_m every timestep.
        #
        # d_m,Γ = P_Γ(d_s - d_s^remesh) depends on the CURRENT solid
        # displacement, which keeps changing after the remesh instant.
        # The zip-based least-squares functional was built once at remesh
        # time (fluid_state['dm_zipped_interface']); we re-solve it here
        # with the latest 'args' to refresh cons['dm'] before every solve.
        # ------------------------------------------------------------
        # NOTE: no per-timestep interface constraint refresh is needed.
        # d_m,Γ = P_Γ(d_s - d_s^remesh) is imposed monolithically by the
        # Lagrange-multiplier residual inside the Newton system, so it is
        # satisfied against the CONVERGED d_s rather than a predictor.

        # ------------------------------------------------------------
        # Solve nonlinear system
        # ------------------------------------------------------------
        try:
            args = system.solve(constrain=cons, arguments=args, tol=1e-9)

        except Exception:
            log.info('Newton failed. Mesh quality of current prediction / last available state:')

            if fluid:
                Jf = function.eval(Jfluid_bz, arguments=args)
                log.info(f'J fluid min/max: {Jf.min():.6e}, {Jf.max():.6e}')

            if solid:
                Js = function.eval(Jsolid_bz, arguments=args)
                dcyl = function.eval(d_cyl_bz, arguments=args)
                dcyl_mag = numpy.linalg.norm(dcyl / 'm', axis=1)

                log.info(f'J solid min/max: {Js.min():.6e}, {Js.max():.6e}')
                log.info(f'particle/interface displacement max: {dcyl_mag.max():.6e} m')

            raise
        # ------------------------------------------------------------
        # Time-step size used for cumulative diagnostics
        # ------------------------------------------------------------
        current_t_s = float(t / 's') if dynamic else 0.0
        dt_s = current_t_s - previous_t_s if dynamic else 0.0
        previous_t_s = current_t_s

        # ------------------------------------------------------------
        # Diagnostics after successful solve
        # ------------------------------------------------------------
        if solid:
            Eeq2_mean_val, Psi_mean_val, Psi_over_G_val = function.eval([Eeq2_mean, Psi_mean, Psi_over_G_mean], arguments=args)

            Eeq_rms = float(numpy.sqrt(Eeq2_mean_val))
            Psi_mean_Pa = float(Psi_mean_val / Pressure('1Pa'))
            Psi_over_G = float(Psi_over_G_val)

            log.info(f'area-averaged equivalent strain: {Eeq_rms:.6e}')
            log.info(f'area-averaged elastic energy density: {Psi_mean_Pa:.6e} Pa')
            log.info(f'area-averaged elastic energy / G: {Psi_over_G:.6e}')
        if fluid:
            Jf = function.eval(Jfluid_bz, arguments=args)
            log.info(f'J fluid min/max: {Jf.min():.6e}, {Jf.max():.6e}')

        if solid:
            Js = function.eval(Jsolid_bz, arguments=args)
            dcyl = function.eval(d_cyl_bz, arguments=args)
            dcyl_mag = numpy.linalg.norm(dcyl / 'm', axis=1)

            log.info(f'J solid min/max: {Js.min():.6e}, {Js.max():.6e}')
            log.info(f'particle/interface displacement max: {dcyl_mag.max():.6e} m')

        # ------------------------------------------------------------
        # Remeshing trigger — checked after every successful solve
        # ------------------------------------------------------------
        remesh_happened = False

        if fluid and solid and enable_remeshing:
            need_remesh = False

            # Quality-based trigger: minimum Jacobian below threshold
            if Jf.min() < remesh_J_threshold:
                log.info(
                    f'Remesh triggered by mesh quality: '
                    f'Jf_min={Jf.min():.4f} < {remesh_J_threshold}'
                )
                need_remesh = True

            # Time-interval-based trigger (optional fallback)
            if remesh_interval_s and current_t_s >= next_remesh_time_s - 0.5 * dt_s:
                log.info(
                    f'Remesh triggered by time interval at t={current_t_s:.6f} s'
                )
                need_remesh = True

            if need_remesh:
                remesh_count += 1

                # Capture old-mesh state for diagnostics BEFORE remeshing.
                # x_bz/u_bz/p_bz already carry consistent definitions
                # (spatial position, TOTAL physical velocity, pressure) both
                # before and after any remesh, so no special-casing is needed.
                xb_current_for_remesh = function.eval(x_bbz, arguments=args) / 'm'
                Jf_old_for_diag       = Jf.copy()
                old_fb_diag  = fluid_bezier   # keep reference to old sampler
                old_x_diag   = function.eval(x_bz, arguments=args) / 'm'
                old_u_diag   = numpy.linalg.norm(
                    function.eval(u_bz, arguments=args) / 'm/s', axis=1)
                old_p_diag   = function.eval(p_bz, arguments=args) / 'Pa'

                (
                    ns, res, cons, ucons, system,
                    fluid_bezier,
                    x_bz, u_bz, p_bz,
                    stagnation_sample, x_stag_sample, u_stag_sample, edot_stag_sample,
                    hydrodynamic_force,
                    Jfluid_bz,
                    fluid_state,
                    args,
                    transfer_metrics,
                ) = remesh_fluid(
                    current_t_s,
                    xb_current_for_remesh,
                    domain,
                    ns,
                    solid,
                    fluid,
                    dynamic,
                    args,
                    topo,
                    fluid_state,
                    remesh_count,
                )

                has_remeshed = True

                # Update remesh schedule
                if remesh_interval_s:
                    next_remesh_time_s = current_t_s + remesh_interval_s

                # Re-evaluate Jf on the fresh mesh
                Jf = function.eval(Jfluid_bz, arguments=args)

                # Capture new-mesh state for diagnostics using the freshly
                # rebuilt x_bz/u_bz/p_bz (already correct on the new mesh).
                new_x_diag  = function.eval(x_bz, arguments=args) / 'm'
                new_u_diag  = numpy.linalg.norm(
                    function.eval(u_bz, arguments=args) / 'm/s', axis=1)
                new_p_diag  = function.eval(p_bz, arguments=args) / 'Pa'

                remesh_diagnostics(
                    current_t_s   = current_t_s,
                    remesh_count  = remesh_count,
                    Jf_old        = Jf_old_for_diag,
                    Jf_new        = Jf,
                    old_fluid_bezier = old_fb_diag,
                    new_fluid_bezier = fluid_bezier,
                    xb_solid_m       = xb_current_for_remesh,
                    old_x_m          = old_x_diag,
                    new_x_m          = new_x_diag,
                    old_u_vals       = old_u_diag,
                    new_u_vals       = new_u_diag,
                    old_p_vals       = old_p_diag,
                    new_p_vals       = new_p_diag,
                    domain           = domain,
                    transfer_metrics = transfer_metrics,
                )

                remesh_happened = True

        if solid:
            # ------------------------------------------------------------
            #  material center and geometric centroid and velocity
            # ------------------------------------------------------------
            xp, yp = function.eval(solid_center_material, arguments=args) / 'm'
            xgeo, ygeo = function.eval(solid_center_geometric, arguments=args) / 'm'
            vxp, vyp = function.eval(solid_velocity_material, arguments=args) / 'm/s'
            Fx, Fy = function.eval(hydrodynamic_force, arguments=args) / 'N/m'

            xb_current = function.eval(x_bbz, arguments=args) / 'm'
            L_particle_m, B_particle_m, deformation_index, orientation_rad = (particle_shape_from_boundary(xb_current))
            a_x = float(numpy.max(xb_current[:, 0]) - numpy.min(xb_current[:, 0]))
            a_y = float(numpy.max(xb_current[:, 1]) - numpy.min(xb_current[:, 1]))
            D_xy = (a_y - a_x) / (a_y + a_x) if (a_y + a_x) > 0.0 else numpy.nan
            particle_effective_radius_m = (0.5 * max(L_particle_m, B_particle_m))

           
            x_search, u_search, edot_search = function.eval([x_stag_sample, u_stag_sample, edot_stag_sample], arguments=args)
            x_search = numpy.asarray(x_search/'m')
            u_search = numpy.asarray(u_search/'m/s')
            edot_search = numpy.asarray(edot_search/'1/s')
            speed_search = numpy.linalg.norm(u_search, axis=1)
            stagnation_search_halfwidth_m = 1.5e-4

            

            junction_mask = ((numpy.abs(x_search[:, 0])<= stagnation_search_halfwidth_m)&(numpy.abs(x_search[:, 1])<= stagnation_search_halfwidth_m))
            distance_to_particle = numpy.hypot(x_search[:, 0] - float(xgeo), x_search[:, 1] - float(ygeo))
            particle_exclusion_mask = (distance_to_particle > 1.2 * particle_effective_radius_m)
            valid_stagnation_points = (junction_mask & particle_exclusion_mask)
            
            if numpy.any(valid_stagnation_points):

                valid_indices = numpy.flatnonzero(valid_stagnation_points)
                local_index = numpy.argmin(speed_search[valid_stagnation_points])
                stagnation_index = valid_indices[local_index]
                x_stagnation_m = float(x_search[stagnation_index, 0])
                y_stagnation_m = float(x_search[stagnation_index, 1])
                minimum_speed_m_s = float(speed_search[stagnation_index])
                edot_at_stagnation = edot_search[stagnation_index] 
                edot_at_stagnation = 0.5 * (edot_at_stagnation+edot_at_stagnation.T)
                principal_strain_rates = numpy.linalg.eigvalsh(edot_at_stagnation)
                epsdot_sp = float(principal_strain_rates[-1])

            else:

                stagnation_index = None
                x_stagnation_m = numpy.nan
                y_stagnation_m = numpy.nan
                minimum_speed_m_s = numpy.nan
                epsdot_sp = numpy.nan
            

            

           


            particle_max_abs_y_m = float(
                numpy.max(numpy.abs(xb_current[:, 1]))
            )
            stop_y_max_m = float(stop_y_max / 'm')
            log.info(
                f'particle maximum y-coordinate: '
                f'{particle_max_abs_y_m:.6e} m '
                f'(stopping value: {stop_y_max_m:.6e} m)'
            )

            distance_from_center_m = numpy.hypot(float(xgeo), float(ygeo))
            residence_flag = int(distance_from_center_m <= residence_radius_m)
            residence_time_s += dt_s * residence_flag

            # ----------------------------------------------------------
            # Probe-point evaluation for continuity monitoring
            # ----------------------------------------------------------
            # Evaluate u and p at fixed spatial probe locations using
            # scattered-data interpolation from the current Gauss sample.
            # This works on any mesh topology without re-building anything.
            probe_sample = stagnation_sample   # already a Gauss-4 sample
            px_all, pu_all, pp_all = function.eval(
                [x_stag_sample, u_stag_sample, p_bz if False else
                 stagnation_sample.bind(ns.p)],
                arguments=args,
            )
            px_all = numpy.asarray(px_all / 'm')
            pu_all = numpy.asarray(pu_all / 'm/s')
            pp_all = numpy.asarray(pp_all / 'Pa')
            pu_mag = numpy.linalg.norm(pu_all, axis=1)

            from scipy.interpolate import LinearNDInterpolator as _LND
            _interp_u = _LND(px_all, pu_mag,  fill_value=numpy.nan)
            _interp_p = _LND(px_all, pp_all,   fill_value=numpy.nan)

            probe_umag = _interp_u(probe_points_m)   # shape (3,)
            probe_p    = _interp_p(probe_points_m)   # shape (3,)

            # ----------------------------------------------------------
            # CSV write
            # ----------------------------------------------------------
            with centroid_csv.open('a') as f:
                f.write(
                    f'{current_t_s:.12e},'
                    f'{istep},'
                    f'{float(xp):.12e},'
                    f'{float(yp):.12e},'
                    f'{float(xgeo):.12e},'
                    f'{float(ygeo):.12e},'
                    f'{float(vxp):.12e},'
                    f'{float(vyp):.12e},'
                    f'{float(Fx):.12e},'
                    f'{float(Fy):.12e},'
                    f'{x_stagnation_m:.12e},'
                    f'{y_stagnation_m:.12e},'
                    f'{minimum_speed_m_s:.12e},'
                    f'{epsdot_sp:.12e},'
                    f'{particle_max_abs_y_m:.12e},'
                    f'{L_particle_m:.12e},'
                    f'{B_particle_m:.12e},'
                    f'{a_x:.12e},'
                    f'{a_y:.12e},'
                    f'{D_xy:.12e},'
                    f'{deformation_index:.12e},'
                    f'{orientation_rad:.12e},'
                    f'{residence_flag},'
                    f'{residence_time_s:.12e},'
                    f'{distance_from_center_m:.12e},'
                    f'{float(Jf.min()):.6e},'
                    f'{float(Jf.max()):.6e},'
                    f'{int(remesh_happened)},'
                    f'{float(probe_umag[0]):.12e},'
                    f'{float(probe_umag[1]):.12e},'
                    f'{float(probe_umag[2]):.12e},'
                    f'{float(probe_p[0]):.12e},'
                    f'{float(probe_p[1]):.12e},'
                    f'{float(probe_p[2]):.12e}\n'
                )

            # log.info(
            #     f'particle maximum y-coordinate: '
            #     f'{particle_max_abs_y_m:.6e} m '
            #     f'(stopping value: {stop_y_max_m:.6e} m)'
            # )
            
            # log.info(
            #     f'particle material center: '
            #     f'x={xp:.6e} m, y={yp:.6e} m'
            # )
            # log.info(
            #     f'particle geometric centroid: '
            #     f'x={xgeo:.6e} m, y={ygeo:.6e} m'
            # )
            # log.info(
            #     f'particle deformation: '
            #     f'L={L_particle_m:.6e} m, B={B_particle_m:.6e} m, '
            #     f'D={deformation_index:.6e}, theta={orientation_rad:.6e} rad'
            # )
            # log.info(
            #     f'residence: flag={residence_flag}, '
            #     f't_res={residence_time_s:.6e} s, '
            #     f'd_center={distance_from_center_m:.6e} m'
            # )

        # ------------------------------------------------------------
        # Evaluate deformed mesh coordinates
        # ------------------------------------------------------------
        x, xb = function.eval([x_bz, x_bbz], arguments=args)

        # ------------------------------------------------------------
        # Save deformed mesh every 100 timesteps
        # ------------------------------------------------------------
        if istep % 100 == 0:

            if solid:
                xs = function.eval(xsolid_bz, arguments=args)

            with export.mplfigure(str('mesh.jpg'), dpi=200) as fig:
                ax = fig.add_subplot(
                    111,
                    title=f'Deformed mesh at t={t:.3s}',
                    ylabel='[m]',
                )

                # Fluid mesh
                if fluid:
                    export.triplot(
                        ax,
                        x / 'm',
                        hull=fluid_bezier.hull,
                        linewidth=0.25,
                    )

                # Solid/particle mesh
                if solid:
                    export.triplot(
                        ax,
                        xs / 'm',
                        hull=solid_bezier.hull,
                        linewidth=0.25,
                    )

                # Particle boundary
                export.triplot(
                    ax,
                    xb / 'm',
                    hull=bbezier.tri,
                    linewidth=1.0,
                    linecolor='b',
                )

                ax.set_aspect('equal')

                L = domain.channel_length / 'm'
                ax.set_xlim(-L, L)
                ax.set_ylim(-L, L)

        # ------------------------------------------------------------
        # Save velocity plot every 100 timesteps
        # ------------------------------------------------------------
        if fluid and istep % 100 == 0:

            u = function.eval(u_bz, arguments=args)
            velmag = numpy.linalg.norm(u / 'm/s', axis=1)

            with export.mplfigure(str('solution.jpg'), dpi=150) as fig:
                ax = fig.add_subplot(
                    111,
                    ylabel='[m]',
                    title=f'Velocity magnitude at t={t:.3s}',
                )

                im = ax.tripcolor(
                    *(x / 'm').T,
                    fluid_bezier.tri,
                    velmag,
                    shading='gouraud',
                )

                fig.colorbar(
                    im,
                    orientation='horizontal',
                    label='velocity [m/s]',
                )

                export.triplot(
                    ax,
                    xb / 'm',
                    hull=bbezier.tri,
                    linewidth=1,
                )

                ax.set_aspect('equal')

                L = domain.channel_length / 'm'
                ax.set_xlim(-L, L)
                ax.set_ylim(-L, L)

        if particle_max_abs_y_m >= stop_y_max_m:
                log.info(
                    f'Stopping simulation: particle maximum y-coordinate '
                    f'reached {particle_max_abs_y_m:.6e} m at '
                    f't={float(t / "s"):.12e} s.'
                )
                break

if __name__ == '__main__':
          cli.choose(main)


