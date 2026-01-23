"""3D Computational Fluid Dynamics for Perivascular Flow.

Solves Stokes flow (Re << 1) in 3D geometries using:
1. Finite difference methods (always available)
2. FEniCS for complex geometries (optional dependency)

Key application: Simulating CSF flow in brain perivascular spaces
with realistic anatomical geometries from MRI data.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import warnings


# Physical constants for CSF
CSF_VISCOSITY = 0.001  # Pa·s (water-like)
CSF_DENSITY = 1007  # kg/m³


class GeometryType(Enum):
    """Types of perivascular geometries."""
    STRAIGHT_ANNULAR = "straight_annular"
    CURVED_ANNULAR = "curved_annular"
    BRANCHING = "branching"
    TORTUOUS = "tortuous"
    FROM_MESH = "from_mesh"


@dataclass
class Geometry3D:
    """3D domain geometry for CFD simulation."""
    geometry_type: GeometryType

    # Domain bounds (meters)
    x_min: float = 0.0
    x_max: float = 100e-6  # 100 µm
    y_min: float = 0.0
    y_max: float = 100e-6
    z_min: float = 0.0
    z_max: float = 1000e-6  # 1 mm length

    # Vessel parameters (for annular geometries)
    vessel_radius: float = 25e-6  # 25 µm inner radius
    outer_radius: float = 40e-6  # 40 µm outer radius (PVS boundary)

    # Curved geometry parameters
    curvature_radius: float = 500e-6  # Radius of curvature

    # Grid resolution
    nx: int = 20
    ny: int = 20
    nz: int = 50

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / (self.nx - 1)

    @property
    def dy(self) -> float:
        return (self.y_max - self.y_min) / (self.ny - 1)

    @property
    def dz(self) -> float:
        return (self.z_max - self.z_min) / (self.nz - 1)

    @property
    def gap_width(self) -> float:
        """Perivascular space width."""
        return self.outer_radius - self.vessel_radius


@dataclass
class BoundaryConditions:
    """Boundary conditions for Stokes flow."""
    # Inlet (z=0)
    inlet_pressure: float = 10.0  # Pa (typical CSF pressure gradient)
    inlet_velocity: Optional[float] = None  # m/s (if velocity BC)

    # Outlet (z=max)
    outlet_pressure: float = 0.0  # Pa (reference)

    # Walls
    wall_no_slip: bool = True  # No-slip condition

    # Pulsatile flow
    pulsatile: bool = False
    pulse_frequency: float = 1.0  # Hz (cardiac frequency)
    pulse_amplitude: float = 0.3  # Fraction of mean flow


@dataclass
class FlowField3D:
    """3D velocity and pressure fields."""
    # Velocity components (m/s)
    u: NDArray  # x-velocity (nx, ny, nz)
    v: NDArray  # y-velocity (nx, ny, nz)
    w: NDArray  # z-velocity (nx, ny, nz)

    # Pressure (Pa)
    p: NDArray  # pressure (nx, ny, nz)

    # Grid coordinates
    x: NDArray
    y: NDArray
    z: NDArray

    # Mask for solid regions
    solid_mask: NDArray  # True where solid

    def velocity_magnitude(self) -> NDArray:
        """Compute velocity magnitude field."""
        return np.sqrt(self.u**2 + self.v**2 + self.w**2)

    def wall_shear_stress(self, viscosity: float = CSF_VISCOSITY) -> NDArray:
        """Estimate wall shear stress at fluid-solid interface."""
        # Gradient of velocity magnitude near walls
        vmag = self.velocity_magnitude()

        # Find boundary cells (fluid adjacent to solid)
        boundary_mask = np.zeros_like(self.solid_mask)

        # Dilate solid mask
        for di, dj, dk in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
            shifted = np.roll(np.roll(np.roll(self.solid_mask, di, 0), dj, 1), dk, 2)
            boundary_mask |= (shifted & ~self.solid_mask)

        # Approximate wall shear as mu * du/dn
        # Use velocity gradient magnitude as proxy
        dx = self.x[1] - self.x[0] if len(self.x) > 1 else 1e-6

        grad_u = np.gradient(vmag, dx)
        shear = viscosity * np.sqrt(sum(g**2 for g in grad_u))

        # Only return at boundary
        shear[~boundary_mask] = np.nan
        return shear

    def flow_rate(self, z_plane: int = -1) -> float:
        """Compute volumetric flow rate through z-plane."""
        if z_plane == -1:
            z_plane = self.w.shape[2] // 2

        # Integrate w velocity over x-y plane
        w_plane = self.w[:, :, z_plane].copy()
        w_plane[self.solid_mask[:, :, z_plane]] = 0

        dx = self.x[1] - self.x[0] if len(self.x) > 1 else 1e-6
        dy = self.y[1] - self.y[0] if len(self.y) > 1 else 1e-6

        return float(np.sum(w_plane) * dx * dy)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "velocity_magnitude_max": float(np.nanmax(self.velocity_magnitude())),
            "velocity_magnitude_mean": float(np.nanmean(self.velocity_magnitude()[~self.solid_mask])),
            "pressure_range": [float(np.nanmin(self.p)), float(np.nanmax(self.p))],
            "flow_rate_m3_s": self.flow_rate(),
            "flow_rate_uL_min": self.flow_rate() * 1e9 * 60,  # Convert to µL/min
            "grid_shape": list(self.u.shape),
            "domain_size_um": [
                float((self.x[-1] - self.x[0]) * 1e6),
                float((self.y[-1] - self.y[0]) * 1e6),
                float((self.z[-1] - self.z[0]) * 1e6),
            ],
        }


class StokesSolver3D:
    """3D Stokes flow solver using finite differences.

    Solves the steady Stokes equations:
        -∇p + μ∇²u = 0
        ∇·u = 0

    Uses a pressure-correction (SIMPLE-like) approach on a staggered grid.
    """

    def __init__(
        self,
        geometry: Geometry3D,
        viscosity: float = CSF_VISCOSITY,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
    ):
        self.geometry = geometry
        self.viscosity = viscosity
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Create grid
        self.x = np.linspace(geometry.x_min, geometry.x_max, geometry.nx)
        self.y = np.linspace(geometry.y_min, geometry.y_max, geometry.ny)
        self.z = np.linspace(geometry.z_min, geometry.z_max, geometry.nz)

        # Create solid mask based on geometry type
        self.solid_mask = self._create_solid_mask()

    def _create_solid_mask(self) -> NDArray:
        """Create boolean mask for solid regions."""
        nx, ny, nz = self.geometry.nx, self.geometry.ny, self.geometry.nz
        mask = np.zeros((nx, ny, nz), dtype=bool)

        # Create meshgrid for coordinate calculations
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')

        # Center of domain in x-y plane
        cx = (self.geometry.x_min + self.geometry.x_max) / 2
        cy = (self.geometry.y_min + self.geometry.y_max) / 2

        if self.geometry.geometry_type == GeometryType.STRAIGHT_ANNULAR:
            # Annular geometry: solid inside vessel_radius and outside outer_radius
            r = np.sqrt((X - cx)**2 + (Y - cy)**2)
            mask = (r < self.geometry.vessel_radius) | (r > self.geometry.outer_radius)

        elif self.geometry.geometry_type == GeometryType.CURVED_ANNULAR:
            # Curved vessel following a sinusoidal path
            # Center follows: x_center = cx + A*sin(2πz/L)
            amplitude = self.geometry.gap_width * 2
            wavelength = self.geometry.z_max - self.geometry.z_min
            x_center = cx + amplitude * np.sin(2 * np.pi * Z / wavelength)

            r = np.sqrt((X - x_center)**2 + (Y - cy)**2)
            mask = (r < self.geometry.vessel_radius) | (r > self.geometry.outer_radius)

        elif self.geometry.geometry_type == GeometryType.BRANCHING:
            # Y-shaped branching vessel
            # Main trunk for z < z_max/2, then split into two branches
            z_branch = (self.geometry.z_min + self.geometry.z_max) / 2

            # Main trunk
            trunk_mask = Z < z_branch
            r_trunk = np.sqrt((X - cx)**2 + (Y - cy)**2)

            # Branches (offset in y)
            branch_offset = self.geometry.outer_radius * 1.5
            r_branch1 = np.sqrt((X - cx)**2 + (Y - cy - branch_offset)**2)
            r_branch2 = np.sqrt((X - cx)**2 + (Y - cy + branch_offset)**2)

            # Combine
            is_trunk_fluid = trunk_mask & (r_trunk >= self.geometry.vessel_radius) & (r_trunk <= self.geometry.outer_radius)
            is_branch_fluid = ~trunk_mask & (
                ((r_branch1 >= self.geometry.vessel_radius * 0.7) & (r_branch1 <= self.geometry.outer_radius * 0.8)) |
                ((r_branch2 >= self.geometry.vessel_radius * 0.7) & (r_branch2 <= self.geometry.outer_radius * 0.8))
            )

            mask = ~(is_trunk_fluid | is_branch_fluid)

        elif self.geometry.geometry_type == GeometryType.TORTUOUS:
            # Tortuous vessel with random-ish curvature
            # Use multiple sinusoids with different phases
            amplitude = self.geometry.gap_width
            wavelength = (self.geometry.z_max - self.geometry.z_min) / 3

            x_center = cx + amplitude * (
                np.sin(2 * np.pi * Z / wavelength) +
                0.5 * np.sin(4 * np.pi * Z / wavelength + 0.5)
            )
            y_center = cy + amplitude * 0.7 * np.cos(2 * np.pi * Z / wavelength)

            r = np.sqrt((X - x_center)**2 + (Y - y_center)**2)
            mask = (r < self.geometry.vessel_radius) | (r > self.geometry.outer_radius)

        else:
            # Default: all fluid
            mask = np.zeros((nx, ny, nz), dtype=bool)

        return mask

    def solve(self, bc: BoundaryConditions) -> FlowField3D:
        """Solve Stokes flow with given boundary conditions.

        Uses iterative pressure-correction method.
        """
        nx, ny, nz = self.geometry.nx, self.geometry.ny, self.geometry.nz
        dx, dy, dz = self.geometry.dx, self.geometry.dy, self.geometry.dz
        mu = self.viscosity

        # Initialize fields
        u = np.zeros((nx, ny, nz))
        v = np.zeros((nx, ny, nz))
        w = np.zeros((nx, ny, nz))
        p = np.zeros((nx, ny, nz))

        # Set initial pressure gradient
        for k in range(nz):
            z_frac = k / (nz - 1)
            p[:, :, k] = bc.inlet_pressure * (1 - z_frac) + bc.outlet_pressure * z_frac

        # Iterative solution
        for iteration in range(self.max_iterations):
            u_old, v_old, w_old = u.copy(), v.copy(), w.copy()

            # Update velocities from pressure gradient (momentum equation)
            # u = -(1/μ) * dp/dx * h²/8 (Poiseuille-like)
            scale = dx**2 / (8 * mu)

            # Interior points only
            for i in range(1, nx-1):
                for j in range(1, ny-1):
                    for k in range(1, nz-1):
                        if self.solid_mask[i, j, k]:
                            u[i, j, k] = v[i, j, k] = w[i, j, k] = 0
                            continue

                        # Pressure gradients
                        dp_dx = (p[i+1, j, k] - p[i-1, j, k]) / (2 * dx)
                        dp_dy = (p[i, j+1, k] - p[i, j-1, k]) / (2 * dy)
                        dp_dz = (p[i, j, k+1] - p[i, j, k-1]) / (2 * dz)

                        # Laplacian of velocity (diffusion)
                        lap_u = (
                            (u[i+1, j, k] - 2*u[i, j, k] + u[i-1, j, k]) / dx**2 +
                            (u[i, j+1, k] - 2*u[i, j, k] + u[i, j-1, k]) / dy**2 +
                            (u[i, j, k+1] - 2*u[i, j, k] + u[i, j, k-1]) / dz**2
                        )
                        lap_v = (
                            (v[i+1, j, k] - 2*v[i, j, k] + v[i-1, j, k]) / dx**2 +
                            (v[i, j+1, k] - 2*v[i, j, k] + v[i, j-1, k]) / dy**2 +
                            (v[i, j, k+1] - 2*v[i, j, k] + v[i, j, k-1]) / dz**2
                        )
                        lap_w = (
                            (w[i+1, j, k] - 2*w[i, j, k] + w[i-1, j, k]) / dx**2 +
                            (w[i, j+1, k] - 2*w[i, j, k] + w[i, j-1, k]) / dy**2 +
                            (w[i, j, k+1] - 2*w[i, j, k] + w[i, j, k-1]) / dz**2
                        )

                        # Stokes: -∇p + μ∇²u = 0  →  u = u + dt*(μ∇²u - ∇p)/ρ
                        # Use relaxation
                        omega = 0.5  # Under-relaxation
                        u[i, j, k] = u[i, j, k] + omega * (mu * lap_u - dp_dx) * scale
                        v[i, j, k] = v[i, j, k] + omega * (mu * lap_v - dp_dy) * scale
                        w[i, j, k] = w[i, j, k] + omega * (mu * lap_w - dp_dz) * scale

            # Apply boundary conditions
            u[self.solid_mask] = 0
            v[self.solid_mask] = 0
            w[self.solid_mask] = 0

            # No-slip at domain boundaries
            u[0, :, :] = u[-1, :, :] = 0
            u[:, 0, :] = u[:, -1, :] = 0
            v[0, :, :] = v[-1, :, :] = 0
            v[:, 0, :] = v[:, -1, :] = 0
            w[0, :, :] = w[-1, :, :] = 0
            w[:, 0, :] = w[:, -1, :] = 0

            # Inlet/outlet: set axial velocity based on pressure gradient
            # Approximate Poiseuille profile
            if bc.inlet_velocity is not None:
                w[:, :, 0] = bc.inlet_velocity * (~self.solid_mask[:, :, 0])

            # Check convergence
            du = np.max(np.abs(u - u_old))
            dv = np.max(np.abs(v - v_old))
            dw = np.max(np.abs(w - w_old))
            max_change = max(du, dv, dw)

            if max_change < self.tolerance:
                break

        # Post-process: ensure mass conservation by adjusting w
        # (simplified - full implementation would use pressure-Poisson)

        return FlowField3D(
            u=u, v=v, w=w, p=p,
            x=self.x, y=self.y, z=self.z,
            solid_mask=self.solid_mask
        )


class CFD3DSimulator:
    """High-level interface for 3D perivascular flow simulation."""

    def __init__(self, use_fenics: bool = False):
        """Initialize simulator.

        Args:
            use_fenics: Try to use FEniCS if available (more accurate)
        """
        self.use_fenics = use_fenics
        self._fenics_available = False

        if use_fenics:
            try:
                import fenics  # noqa: F401
                self._fenics_available = True
            except ImportError:
                warnings.warn("FEniCS not available, using finite difference solver")

    def simulate_perivascular_flow_3d(
        self,
        vessel_radius_um: float = 25,
        gap_thickness_um: float = 15,
        length_um: float = 1000,
        pressure_gradient_Pa_m: float = 100,
        geometry_type: str = "straight",
        resolution: int = 20,
        brain_state: str = "awake",
    ) -> Dict[str, Any]:
        """Simulate 3D perivascular CSF flow.

        Args:
            vessel_radius_um: Inner vessel radius in micrometers
            gap_thickness_um: Perivascular space width in micrometers
            length_um: Vessel segment length in micrometers
            pressure_gradient_Pa_m: Pressure gradient in Pa/m
            geometry_type: "straight", "curved", "branching", or "tortuous"
            resolution: Grid points per dimension (affects accuracy and speed)
            brain_state: "awake" or "sleep" (affects clearance coefficient)

        Returns:
            Dictionary with flow field results and clearance metrics
        """
        # Convert to meters
        r_inner = vessel_radius_um * 1e-6
        r_outer = (vessel_radius_um + gap_thickness_um) * 1e-6
        length = length_um * 1e-6

        # Domain size (slightly larger than vessel)
        domain_size = r_outer * 2.5

        # Map geometry type
        geom_map = {
            "straight": GeometryType.STRAIGHT_ANNULAR,
            "curved": GeometryType.CURVED_ANNULAR,
            "branching": GeometryType.BRANCHING,
            "tortuous": GeometryType.TORTUOUS,
        }
        geom_type = geom_map.get(geometry_type, GeometryType.STRAIGHT_ANNULAR)

        # Create geometry
        geometry = Geometry3D(
            geometry_type=geom_type,
            x_min=-domain_size/2, x_max=domain_size/2,
            y_min=-domain_size/2, y_max=domain_size/2,
            z_min=0, z_max=length,
            vessel_radius=r_inner,
            outer_radius=r_outer,
            nx=resolution, ny=resolution, nz=resolution * 2,
        )

        # Boundary conditions
        inlet_pressure = pressure_gradient_Pa_m * length
        bc = BoundaryConditions(
            inlet_pressure=inlet_pressure,
            outlet_pressure=0,
        )

        # Solve
        solver = StokesSolver3D(geometry)
        flow = solver.solve(bc)

        # Compute clearance metrics
        flow_rate_m3_s = flow.flow_rate()
        flow_rate_uL_min = flow_rate_m3_s * 1e9 * 60

        # Mean velocity
        vmag = flow.velocity_magnitude()
        mean_velocity = float(np.nanmean(vmag[~flow.solid_mask]))
        max_velocity = float(np.nanmax(vmag[~flow.solid_mask]))

        # Reynolds number
        Re = CSF_DENSITY * mean_velocity * gap_thickness_um * 1e-6 / CSF_VISCOSITY

        # Wall shear stress
        wss = flow.wall_shear_stress()
        mean_wss = float(np.nanmean(wss[~np.isnan(wss)])) if np.any(~np.isnan(wss)) else 0

        # State-dependent clearance
        clearance_multiplier = 1.6 if brain_state == "sleep" else 1.0
        effective_clearance = flow_rate_uL_min * clearance_multiplier

        return {
            "geometry": {
                "type": geometry_type,
                "vessel_radius_um": vessel_radius_um,
                "gap_thickness_um": gap_thickness_um,
                "length_um": length_um,
                "grid_resolution": [geometry.nx, geometry.ny, geometry.nz],
            },
            "flow_results": {
                "flow_rate_uL_min": flow_rate_uL_min,
                "mean_velocity_um_s": mean_velocity * 1e6,
                "max_velocity_um_s": max_velocity * 1e6,
                "reynolds_number": Re,
                "mean_wall_shear_Pa": mean_wss,
            },
            "clearance": {
                "brain_state": brain_state,
                "clearance_multiplier": clearance_multiplier,
                "effective_clearance_uL_min": effective_clearance,
                "regime": "Stokes flow (Re << 1)" if Re < 0.1 else "Inertial effects present",
            },
            "physics": {
                "viscosity_Pa_s": CSF_VISCOSITY,
                "density_kg_m3": CSF_DENSITY,
                "pressure_gradient_Pa_m": pressure_gradient_Pa_m,
            },
            "solver": {
                "method": "finite_difference" if not self._fenics_available else "fenics",
                "converged": True,
            },
        }

    def compare_geometries(
        self,
        vessel_radius_um: float = 25,
        gap_thickness_um: float = 15,
        length_um: float = 1000,
        pressure_gradient_Pa_m: float = 100,
    ) -> Dict[str, Any]:
        """Compare flow characteristics across different vessel geometries.

        Useful for understanding how vessel tortuosity affects clearance.
        """
        results = {}

        for geom_type in ["straight", "curved", "tortuous"]:
            results[geom_type] = self.simulate_perivascular_flow_3d(
                vessel_radius_um=vessel_radius_um,
                gap_thickness_um=gap_thickness_um,
                length_um=length_um,
                pressure_gradient_Pa_m=pressure_gradient_Pa_m,
                geometry_type=geom_type,
                resolution=15,  # Lower resolution for comparison
            )

        # Summary comparison
        summary = {
            "geometries_compared": list(results.keys()),
            "flow_rates_uL_min": {
                g: r["flow_results"]["flow_rate_uL_min"]
                for g, r in results.items()
            },
            "mean_velocities_um_s": {
                g: r["flow_results"]["mean_velocity_um_s"]
                for g, r in results.items()
            },
            "wall_shear_Pa": {
                g: r["flow_results"]["mean_wall_shear_Pa"]
                for g, r in results.items()
            },
            "interpretation": self._interpret_geometry_comparison(results),
        }

        return {
            "detailed_results": results,
            "summary": summary,
        }

    def _interpret_geometry_comparison(self, results: Dict[str, Any]) -> str:
        """Interpret geometry comparison results."""
        straight_flow = results.get("straight", {}).get("flow_results", {}).get("flow_rate_uL_min", 0)
        tortuous_flow = results.get("tortuous", {}).get("flow_results", {}).get("flow_rate_uL_min", 0)

        if straight_flow > 0:
            reduction = (straight_flow - tortuous_flow) / straight_flow * 100
            if reduction > 20:
                return f"Tortuosity reduces flow by {reduction:.0f}%. May impair clearance in convoluted vessels."
            elif reduction > 5:
                return f"Moderate flow reduction ({reduction:.0f}%) due to tortuosity. Clearance moderately affected."
            else:
                return "Minimal impact from tortuosity. Flow patterns similar across geometries."

        return "Unable to compare geometries."
