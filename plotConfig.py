from typing import Optional
from dataclasses import dataclass

@dataclass
class PlotConfig:
    """Configuration for individual MRI plotting."""
    padding: int = 10
    fps: int = 10
    crop: bool = False
    reorient: bool = True
    mask_underlay: bool = False
    mask: Optional[str] = None
    underlay_image: Optional[str] = None
    dpi: int = 100
    figure_scale_factor: int = 6
    max_fig_size = 1500
    displacement_scale_factor: int = 120
    displacement_downsample_factor: int = 5
    displacement_arrow_thickness: float = 0.003