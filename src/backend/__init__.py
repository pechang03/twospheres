"""TwoSphere backend modules.

Organized by functor level:
- optics: F₀-F₂ core physics
- mri: F₀-F₂ MRI geometry
- services: F₃-F₄ composed services
- visualize: F₁-F₂ plotting
"""

from . import optics
from . import mri
from . import services
from . import visualize

__all__ = ['optics', 'mri', 'services', 'visualize']
