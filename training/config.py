from dataclasses import dataclass, field


@dataclass
class Material:
    name: str = ""
    E: float = 114_000.0        # MPa
    nu: float = 0.34
    yield_strength: float = 880.0  # MPa
    density: float = 4.43e-3    # g/mm^3


@dataclass
class LatticeModel:
    name: str = "BCC"
    rel_density: float = 0.30
    c_E: float = 1.0
    n_E: float = 2.0
    c_sy: float = 0.3
    n_sy: float = 1.5
    nu: float = 0.33


@dataclass
class Hole:
    y: float
    z: float
    r: float
    x_min: float
    x_max: float


@dataclass
class Case:
    holes: list = field(default_factory=lambda: [
        Hole(10, 15, 4, -1, 16), Hole(70, 15, 4, -1, 16),
        Hole(10, 65, 4, -1, 16), Hole(70, 65, 4, -1, 16),
    ])
    load_x: float = 110.0
    force: tuple = (0.0, 0.0, -5000.0)  
    mesh_size: float = 3.0              
    shell_thickness: float = 3.0        
    boss_thickness: float = 4.0         
