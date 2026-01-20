"""Ray trace visualization for LOC photonic devices.

F₁-F₂ level: Generate PNG diagrams from pyoptools ray traces.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path


def draw_plano_convex_lens(
    ax,
    z_center: float,
    radius: float,
    thickness: float,
    lens_diameter: float,
    orientation: str = 'right',
    color: str = 'deepskyblue',
    label: Optional[str] = None
):
    """Draw a plano-convex lens with proper curved surface.
    
    Args:
        ax: Matplotlib axes
        z_center: Z position of lens center
        radius: Radius of curvature (mm)
        thickness: Lens thickness (mm)
        lens_diameter: Lens diameter (mm)
        orientation: 'right' = curved on right, 'left' = curved on left
        color: Fill color
        label: Optional label text
    """
    z_flat = z_center - thickness/2 if orientation == 'right' else z_center + thickness/2
    z_curved = z_center + thickness/2 if orientation == 'right' else z_center - thickness/2
    
    # Flat side
    ax.plot([z_flat, z_flat], [-lens_diameter/2, lens_diameter/2], 'b-', linewidth=1.5)
    
    # Curved side (arc)
    h = np.linspace(-lens_diameter/2, lens_diameter/2, 50)
    sag = radius - np.sqrt(np.maximum(0, radius**2 - h**2))
    
    if orientation == 'right':
        z_curve = z_curved - sag
        ax.fill_betweenx(h, z_flat, z_curve, alpha=0.4, color=color)
    else:
        z_curve = z_curved + sag
        ax.fill_betweenx(h, z_curve, z_flat, alpha=0.4, color=color)
    
    ax.plot(z_curve, h, 'b-', linewidth=1.5)
    
    if label:
        ax.text(z_center, lens_diameter/2 + 0.15, label, 
                ha='center', va='bottom', fontsize=7)


def extract_ray_path(ray, z_offset: float = 0) -> Tuple[List[float], List[float]]:
    """Extract Z and Y coordinates from a pyoptools ray.
    
    Args:
        ray: pyoptools Ray object
        z_offset: Offset to add to Z coordinates
        
    Returns:
        Tuple of (z_positions, y_positions)
    """
    z_pts = []
    y_pts = []
    
    current = ray
    z_pts.append(current.origin[2] + z_offset)
    y_pts.append(current.origin[1])
    
    while current.childs:
        current = current.childs[0]
        z_pts.append(current.origin[2] + z_offset)
        y_pts.append(current.origin[1])
    
    return z_pts, y_pts


def plot_phooc_system(
    rays: List,
    lenses: List[Dict[str, Any]],
    chamber: Optional[Dict[str, Any]] = None,
    fibers: Optional[Dict[str, Any]] = None,
    chip_size: Tuple[float, float] = (10.0, 5.0),
    title: str = 'PhOOC System',
    output_path: Optional[str] = None,
    z_offset: float = 0,
    figsize: Tuple[float, float] = (14, 6),
    dpi: int = 150
) -> Optional[str]:
    """Generate a PhOOC ray trace diagram.
    
    Args:
        rays: List of pyoptools Ray objects (after propagation)
        lenses: List of lens specs: [{'z': z_pos, 'R': radius, 't': thickness, 
                                      'd': diameter, 'orientation': 'right'/'left',
                                      'label': 'L1'}]
        chamber: Chamber spec: {'z_start': z, 'size': (w, h)}
        fibers: Fiber spec: {'input': {'z': z, 'core_um': 50}, 
                             'output': {'z': z, 'core_um': 150}}
        chip_size: (length_mm, width_mm)
        title: Plot title
        output_path: Path to save PNG (None = don't save)
        z_offset: Offset for ray Z coordinates
        figsize: Figure size in inches
        dpi: Resolution
        
    Returns:
        Output path if saved, None otherwise
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    chip_length, chip_width = chip_size
    
    # Draw chip outline
    chip = patches.Rectangle(
        (0, -chip_width/2), chip_length, chip_width,
        linewidth=2, edgecolor='darkblue', facecolor='lightcyan', alpha=0.3
    )
    ax.add_patch(chip)
    ax.text(chip_length/2, chip_width/2 + 0.2, f'PDMS PhOOC ({chip_length}mm)',
            ha='center', fontsize=10, fontweight='bold')
    
    # Draw lenses
    for lens in lenses:
        draw_plano_convex_lens(
            ax,
            z_center=lens['z'],
            radius=lens['R'],
            thickness=lens.get('t', 0.5),
            lens_diameter=lens.get('d', 1.5),
            orientation=lens.get('orientation', 'right'),
            color=lens.get('color', 'deepskyblue'),
            label=lens.get('label')
        )
    
    # Draw chamber
    if chamber:
        z_start = chamber['z_start']
        w, h = chamber.get('size', (1.0, 1.0))
        chamber_rect = patches.Rectangle(
            (z_start, -h/2), w, h,
            linewidth=2, edgecolor='darkgreen', facecolor='lightgreen', alpha=0.5
        )
        ax.add_patch(chamber_rect)
        ax.text(z_start + w/2, -h/2 - 0.15, 'Chamber',
                ha='center', va='top', fontsize=8, color='darkgreen')
    
    # Draw fibers
    if fibers:
        if 'input' in fibers:
            inp = fibers['input']
            ax.plot([-0.8, inp['z']], [0, 0], 'orange', 
                    linewidth=max(1, inp.get('core_um', 50)/50))
            ax.text(-0.5, 0.35, f"Input\n{inp.get('core_um', '?')}µm",
                    ha='center', fontsize=7, color='darkorange')
        
        if 'output' in fibers:
            out = fibers['output']
            ax.plot([out['z'], chip_length + 0.8], [0, 0], 'darkorange',
                    linewidth=max(1, out.get('core_um', 50)/50))
            ax.text(chip_length + 0.5, 0.35, f"Output\n{out.get('core_um', '?')}µm",
                    ha='center', fontsize=7, color='darkorange')
    
    # Draw rays
    for ray in rays:
        z_pts, y_pts = extract_ray_path(ray, z_offset)
        ax.plot(z_pts, y_pts, 'r-', linewidth=0.5, alpha=0.6)
    
    # Labels
    ax.set_xlabel('Z position (mm)', fontsize=10)
    ax.set_ylabel('Y position (mm)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim(-1.5, chip_length + 1.5)
    ax.set_ylim(-chip_width/2 - 0.8, chip_width/2 + 0.8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi)
        plt.close(fig)
        return output_path
    
    plt.close(fig)
    return None


def plot_ring_resonator(
    radius_um: float,
    coupling_gap_um: float = 200,
    bus_width_um: float = 50,
    output_path: Optional[str] = None,
    title: str = 'Ring Resonator',
    dpi: int = 150
) -> Optional[str]:
    """Generate a ring resonator schematic.
    
    Args:
        radius_um: Ring radius in µm
        coupling_gap_um: Gap between ring and bus waveguide in µm
        bus_width_um: Bus waveguide width in µm
        output_path: Path to save PNG
        title: Plot title
        dpi: Resolution
        
    Returns:
        Output path if saved
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scale to mm for plotting
    r = radius_um / 1000
    gap = coupling_gap_um / 1000
    bus_w = bus_width_um / 1000
    
    # Draw ring
    ring = patches.Circle((0, r + gap + bus_w/2), r, 
                          fill=False, edgecolor='blue', linewidth=2)
    ax.add_patch(ring)
    
    # Draw bus waveguide
    bus = patches.Rectangle((-r*1.5, -bus_w/2), r*3, bus_w,
                           facecolor='lightblue', edgecolor='blue', linewidth=1)
    ax.add_patch(bus)
    
    # Arrows for light direction
    ax.annotate('', xy=(-r*1.2, 0), xytext=(-r*1.5, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.annotate('', xy=(r*1.5, 0), xytext=(r*1.2, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    # Labels
    ax.text(0, r + gap + bus_w/2 + r + 0.05, f'R = {radius_um:.0f} µm',
            ha='center', fontsize=10)
    ax.text(0, -bus_w/2 - 0.05, f'Gap = {coupling_gap_um:.0f} nm',
            ha='center', va='top', fontsize=9)
    
    ax.set_xlim(-r*2, r*2)
    ax.set_ylim(-r*0.5, r*2.5 + gap)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return output_path
    
    plt.close(fig)
    return None
