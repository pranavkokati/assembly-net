"""
Physical Constants and Material-Specific Parameters.

This module provides physically grounded constants for coordination chemistry
simulations, with references to experimental literature.

References
----------
[1] Ejima, H. et al. (2013). One-step assembly of coordination complexes for
    versatile film and particle engineering. Science 341(6142), 154-157.
[2] Guo, J. et al. (2014). Engineering multifunctional capsules through the
    assembly of metal-phenolic networks. Angew. Chem. 126(22), 5652-5657.
[3] Rahim, M. A. et al. (2019). Metal-phenolic supramolecular gelation.
    Angew. Chem. 131(14), 4584-4592.
[4] Smith, D. W. (2006). Ionic hydration enthalpies. J. Chem. Educ. 54(9), 540.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict


# =============================================================================
# FUNDAMENTAL CONSTANTS
# =============================================================================

@dataclass(frozen=True)
class PhysicalConstants:
    """Fundamental physical constants (SI units)."""

    # Boltzmann constant (J/K)
    k_B: float = 1.380649e-23

    # Planck constant (J·s)
    h: float = 6.62607015e-34

    # Gas constant (J/mol·K)
    R: float = 8.314462618

    # Avogadro constant (mol^-1)
    N_A: float = 6.02214076e23

    # Elementary charge (C)
    e: float = 1.602176634e-19

    # Vacuum permittivity (F/m)
    epsilon_0: float = 8.8541878128e-12


# =============================================================================
# METAL ION PROPERTIES
# =============================================================================

class MetalIon(Enum):
    """Common metal ions in coordination networks."""
    FE3 = "Fe(III)"
    FE2 = "Fe(II)"
    CU2 = "Cu(II)"
    ZN2 = "Zn(II)"
    CO2 = "Co(II)"
    NI2 = "Ni(II)"
    AL3 = "Al(III)"
    CA2 = "Ca(II)"
    MG2 = "Mg(II)"


@dataclass
class MetalProperties:
    """
    Properties of metal ions relevant to coordination chemistry.

    Attributes
    ----------
    symbol : str
        Chemical symbol with oxidation state.
    coordination_number : int
        Preferred coordination number.
    ionic_radius : float
        Ionic radius in pm (picometers).
    charge : int
        Formal charge.
    hard_soft : str
        Hard/soft acid classification (HSAB theory).
    typical_bond_energy : float
        Typical M-O bond energy in kJ/mol.

    References
    ----------
    [4] Smith, D. W. (2006). Ionic hydration enthalpies. J. Chem. Educ.
    Shannon, R. D. (1976). Acta Cryst. A32, 751.
    """
    symbol: str
    coordination_number: int
    ionic_radius: float  # pm
    charge: int
    hard_soft: str  # "hard", "intermediate", "soft"
    typical_bond_energy: float  # kJ/mol (M-O bond)


# Metal ion database
METAL_PROPERTIES: Dict[MetalIon, MetalProperties] = {
    MetalIon.FE3: MetalProperties(
        symbol="Fe(III)",
        coordination_number=6,
        ionic_radius=65,
        charge=3,
        hard_soft="hard",
        typical_bond_energy=400,  # Fe-O bond ~400 kJ/mol
    ),
    MetalIon.FE2: MetalProperties(
        symbol="Fe(II)",
        coordination_number=6,
        ionic_radius=78,
        charge=2,
        hard_soft="intermediate",
        typical_bond_energy=350,
    ),
    MetalIon.CU2: MetalProperties(
        symbol="Cu(II)",
        coordination_number=4,  # Often square planar
        ionic_radius=73,
        charge=2,
        hard_soft="intermediate",
        typical_bond_energy=340,
    ),
    MetalIon.ZN2: MetalProperties(
        symbol="Zn(II)",
        coordination_number=4,  # Often tetrahedral
        ionic_radius=74,
        charge=2,
        hard_soft="intermediate",
        typical_bond_energy=280,
    ),
    MetalIon.AL3: MetalProperties(
        symbol="Al(III)",
        coordination_number=6,
        ionic_radius=54,
        charge=3,
        hard_soft="hard",
        typical_bond_energy=500,
    ),
    MetalIon.CA2: MetalProperties(
        symbol="Ca(II)",
        coordination_number=8,
        ionic_radius=100,
        charge=2,
        hard_soft="hard",
        typical_bond_energy=200,
    ),
}


# =============================================================================
# LIGAND PROPERTIES
# =============================================================================

class LigandType(Enum):
    """Common ligand types in metal-phenolic networks."""
    CATECHOL = "catechol"
    GALLOL = "gallol"
    TANNIC_ACID = "tannic_acid"
    DOPAMINE = "dopamine"
    CARBOXYLATE = "carboxylate"


@dataclass
class LigandProperties:
    """
    Properties of ligands relevant to coordination chemistry.

    Attributes
    ----------
    name : str
        Common name.
    denticity : int
        Number of coordination sites.
    pKa_values : list
        pKa values of coordinating groups.
    molecular_weight : float
        Molecular weight in g/mol.

    References
    ----------
    [1] Ejima et al. (2013). Science 341, 154-157.
    """
    name: str
    denticity: int
    pKa_values: list  # pKa of coordinating groups
    molecular_weight: float  # g/mol


LIGAND_PROPERTIES: Dict[LigandType, LigandProperties] = {
    LigandType.CATECHOL: LigandProperties(
        name="catechol",
        denticity=2,
        pKa_values=[9.4, 13.0],  # Two phenolic OH groups
        molecular_weight=110.1,
    ),
    LigandType.GALLOL: LigandProperties(
        name="gallic acid / gallol",
        denticity=3,  # Can be tridentate
        pKa_values=[8.7, 11.4, 13.1],
        molecular_weight=170.1,
    ),
    LigandType.TANNIC_ACID: LigandProperties(
        name="tannic acid",
        denticity=20,  # Multiple gallol groups
        pKa_values=[8.5],  # Approximate average
        molecular_weight=1701.2,
    ),
    LigandType.DOPAMINE: LigandProperties(
        name="dopamine",
        denticity=2,
        pKa_values=[9.0, 10.6],
        molecular_weight=153.2,
    ),
}


# =============================================================================
# KINETIC PARAMETERS
# =============================================================================

@dataclass
class KineticParameters:
    """
    Kinetic parameters for coordination bond formation/dissociation.

    Based on experimental data for metal-phenolic systems.

    Attributes
    ----------
    delta_G_formation : float
        Activation free energy for bond formation (kJ/mol).
    delta_G_dissociation : float
        Activation free energy for bond dissociation (kJ/mol).
    ph_dependence : float
        Sensitivity to pH (slope of log(k) vs pH).

    References
    ----------
    [1] Ejima et al. (2013). Assembly kinetics ~seconds at pH 7.4.
    [3] Rahim et al. (2019). Gelation kinetics in metal-phenolic systems.
    """
    delta_G_formation: float  # kJ/mol
    delta_G_dissociation: float  # kJ/mol
    ph_dependence: float  # d(log k)/d(pH)


# Default kinetic parameters for Fe(III)-catechol at 25°C
DEFAULT_KINETICS = KineticParameters(
    delta_G_formation=50.0,   # ~50 kJ/mol for fast assembly
    delta_G_dissociation=75.0,  # ~75 kJ/mol (bonds are relatively stable)
    ph_dependence=0.5,  # Moderate pH dependence
)


# =============================================================================
# ASSEMBLY REGIME PARAMETERS
# =============================================================================

@dataclass
class AssemblyRegimeParameters:
    """
    Parameters characterizing different assembly regimes.

    DLA (Diffusion-Limited Aggregation):
        - Fast bond formation relative to diffusion
        - Results in branching, fractal-like structures
        - Early loop formation as branches connect

    RLA (Reaction-Limited Aggregation):
        - Slow bond formation relative to diffusion
        - Particles collide many times before bonding
        - More compact, reorganized structures
        - Late loop formation after reorganization

    Burst Nucleation:
        - Initial rapid nucleation burst
        - Followed by slow growth
        - Many small clusters that gradually merge

    References
    ----------
    Meakin, P. (1998). Fractals, scaling and growth far from equilibrium.
    Cambridge University Press.
    """
    name: str
    formation_rate_multiplier: float
    dissociation_rate_multiplier: float
    description: str


ASSEMBLY_REGIMES = {
    "DLA": AssemblyRegimeParameters(
        name="Diffusion-Limited Aggregation",
        formation_rate_multiplier=5.0,
        dissociation_rate_multiplier=0.1,
        description="Fast, irreversible bonding. Branched structures. Early loops.",
    ),
    "RLA": AssemblyRegimeParameters(
        name="Reaction-Limited Aggregation",
        formation_rate_multiplier=0.5,
        dissociation_rate_multiplier=1.0,
        description="Slow bonding with reorganization. Compact structures. Late loops.",
    ),
    "BURST": AssemblyRegimeParameters(
        name="Burst Nucleation",
        formation_rate_multiplier=2.0,  # Initial burst
        dissociation_rate_multiplier=0.3,
        description="Rapid initial nucleation, then slow growth. Many small clusters.",
    ),
}
