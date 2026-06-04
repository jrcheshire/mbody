"""Cloud-in-cell (CIC) mass assignment: particles <-> mesh, differentiably.

The PM method constantly moves between two representations: particles (which
carry the dynamics) and a regular mesh (where the FFT force solve happens). CIC
is the standard bridge.

**Paint** (particles -> mesh): each particle is treated as a unit cube of mass
("cloud") the size of one cell; it deposits into the 8 nearest grid cells with
trilinear weights (the fractional cube overlap). The weights along each axis are
(1 - f) and f for the lower/upper cell, where f is the particle's fractional
position in its cell, so the 8 weights sum to 1 and mass is conserved exactly.

**Read** (mesh -> particles): the transpose operation -- trilinearly
interpolate a mesh field (e.g. the force) back to particle positions with the
same 8 weights. Paint and read are exact adjoints, which matters for gradients.

Why this matters for M-body: the whole point is differentiability, and CIC is
the one nonlinear-looking step besides the FFT. The fractional weights f depend
smoothly on particle position (the cell *index* does not, but its gradient is
zero), so `mx.grad` flows through painting to the particle positions -- the
scatter-add primitive we verified in Phase 0. This is what lets a gradient
reach back through "where the particles landed."
"""

import mlx.core as mx

from mbody import precision as P

# The 8 cell corners of a CIC cloud, as (dx, dy, dz) offsets.
_CORNERS = [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]


def _cic_pieces(positions, box):
    """Shared CIC setup: integer base cell, fractional offset, cell size."""
    N, L = box.n_mesh, box.box_size
    d = L / N
    xp = positions / d  # positions in cell units
    base_f = mx.floor(xp)
    frac = xp - base_f  # in [0, 1); floor has zero gradient, so d frac/dx = 1
    # Detach the integer cell indices: gradients flow through the CIC weights
    # (frac), never through which cell a particle lands in (that map is
    # piecewise-constant). MLX's scatter/gather also refuse a VJP with respect
    # to index arrays, so stop_gradient here is required for mx.grad to work.
    base = mx.stop_gradient(base_f).astype(mx.int32)
    return N, base, frac


def cic_paint(positions, box, weights=None):
    """Paint particles onto the mesh by CIC. Returns a real (N, N, N) field.

    With unit weights the field is a number-count per cell whose total equals
    the particle count (mass is conserved). Differentiable in `positions` and
    `weights`.
    """
    N, base, frac = _cic_pieces(positions, box)
    n_part = positions.shape[0]
    if weights is None:
        weights = mx.ones((n_part,), dtype=P.REAL)

    wlo = 1.0 - frac
    mesh = mx.zeros((N * N * N,), dtype=P.REAL)
    for dx, dy, dz in _CORNERS:
        wx = frac[:, 0] if dx else wlo[:, 0]
        wy = frac[:, 1] if dy else wlo[:, 1]
        wz = frac[:, 2] if dz else wlo[:, 2]
        w = wx * wy * wz * weights
        ix = (base[:, 0] + dx) % N
        iy = (base[:, 1] + dy) % N
        iz = (base[:, 2] + dz) % N
        flat = (ix * N + iy) * N + iz
        mesh = mesh.at[flat].add(w)
    return mesh.reshape(N, N, N)


def density_contrast(positions, box):
    """CIC density contrast delta = rho/mean - 1 from particle positions."""
    N = box.n_mesh
    rho = cic_paint(positions, box)
    mean = positions.shape[0] / (N**3)
    return rho / mean - 1.0


def cic_read(field, positions, box):
    """Trilinearly interpolate a mesh `field` to particle positions (the CIC
    adjoint of paint). Returns a (n_particles,) array, differentiable in both
    `field` and `positions`.
    """
    N, base, frac = _cic_pieces(positions, box)
    flat_field = field.reshape(-1)
    wlo = 1.0 - frac
    out = mx.zeros((positions.shape[0],), dtype=P.REAL)
    for dx, dy, dz in _CORNERS:
        wx = frac[:, 0] if dx else wlo[:, 0]
        wy = frac[:, 1] if dy else wlo[:, 1]
        wz = frac[:, 2] if dz else wlo[:, 2]
        w = wx * wy * wz
        ix = (base[:, 0] + dx) % N
        iy = (base[:, 1] + dy) % N
        iz = (base[:, 2] + dz) % N
        flat = (ix * N + iy) * N + iz
        out = out + w * flat_field[flat]
    return out
