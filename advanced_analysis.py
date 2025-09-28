"""
Advanced Analysis Capabilities for Industry-Ready Aircraft Design Tool
Implements sophisticated aerodynamic and structural analysis methods beyond basic calculations
"""

import numpy as np
import scipy as sp
from scipy import interpolate, optimize, integrate
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import math
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

from validation import handle_analysis_errors, AnalysisError, PhysicsError, logger

@dataclass
class CompressibilityCorrection:
    """Compressibility correction results"""
    mach_number: float
    correction_factor: float
    method: str
    validity_range: str
    
@dataclass
class CompositeFailureAnalysis:
    """Composite material failure analysis results"""
    tsai_wu_index: float
    hashin_fiber_failure: float
    hashin_matrix_failure: float
    first_ply_failure_load: float
    failure_mode: str
    margin_of_safety: float

class AdvancedAerodynamics:
    """Advanced aerodynamic analysis beyond basic panel methods"""
    
    @staticmethod
    @handle_analysis_errors
    def prandtl_glauert_correction(cl_incompressible: float, mach: float) -> CompressibilityCorrection:
        """
        Prandtl-Glauert compressibility correction for subsonic flow
        Valid for M < 0.8 approximately
        """
        if mach >= 1.0:
            raise PhysicsError("Prandtl-Glauert correction invalid for supersonic flow")
        
        if mach >= 0.8:
            warnings.warn("Prandtl-Glauert correction approaching validity limit at M > 0.8")
        
        beta = math.sqrt(1 - mach**2)
        correction_factor = 1 / beta
        cl_corrected = cl_incompressible * correction_factor
        
        return CompressibilityCorrection(
            mach_number=mach,
            correction_factor=correction_factor,
            method="Prandtl-Glauert",
            validity_range="M < 0.8"
        ), cl_corrected
    
    @staticmethod
    @handle_analysis_errors
    def karman_tsien_correction(cl_incompressible: float, mach: float) -> CompressibilityCorrection:
        """
        Kármán-Tsien compressibility correction - more accurate than Prandtl-Glauert
        Valid for M < 0.9 approximately
        """
        if mach >= 1.0:
            raise PhysicsError("Kármán-Tsien correction invalid for supersonic flow")
        
        beta = math.sqrt(1 - mach**2)
        gamma = 1.4  # Specific heat ratio for air
        
        # Kármán-Tsien correction factor
        correction_factor = 1 / (beta + (mach**2 / (1 + beta)) * cl_incompressible / 2)
        cl_corrected = cl_incompressible * correction_factor
        
        return CompressibilityCorrection(
            mach_number=mach,
            correction_factor=correction_factor,
            method="Kármán-Tsien",
            validity_range="M < 0.9"
        ), cl_corrected
    
    @staticmethod
    @handle_analysis_errors
    def critical_mach_number(cl: float, thickness_ratio: float) -> float:
        """
        Calculate critical Mach number where local sonic flow first occurs
        """
        # Simplified critical Mach calculation using empirical correlations
        # Based on NACA data and experimental results
        
        # Base critical Mach number for symmetric airfoil
        m_crit_base = 0.87 - 0.17 * thickness_ratio
        
        # Correction for lift coefficient (higher lift reduces critical Mach)
        lift_correction = 0.1 * abs(cl)
        
        # Apply thickness penalty for thick airfoils
        if thickness_ratio > 0.05:
            thickness_penalty = 0.1 * (thickness_ratio - 0.05)**2
        else:
            thickness_penalty = 0.0
        
        # Calculate critical Mach number
        m_crit = m_crit_base - lift_correction - thickness_penalty
        
        # Ensure reasonable bounds
        m_crit = max(0.5, min(0.9, m_crit))
        
        return m_crit
    
    @staticmethod
    @handle_analysis_errors
    def wave_drag_estimation(mach: float, thickness_ratio: float, sweep_angle: float = 0.0) -> float:
        """
        Estimate wave drag for transonic/supersonic conditions
        Uses simplified area rule and shock-expansion theory
        """
        if mach <= 1.0:
            return 0.0  # No wave drag in subsonic flow
        
        # Simplified wave drag coefficient estimation
        # Based on linearized supersonic theory
        
        sweep_rad = math.radians(sweep_angle)
        mach_normal = mach * math.cos(sweep_rad)
        
        if mach_normal <= 1.0:
            # Subsonic normal component
            return 0.0
        
        # Shock wave strength parameter
        beta = math.sqrt(mach_normal**2 - 1)
        
        # Wave drag coefficient (simplified)
        cd_wave = 4 * thickness_ratio**2 / beta
        
        return cd_wave
    
    @staticmethod
    @handle_analysis_errors
    def viscous_inviscid_coupling(inviscid_cp: np.ndarray, reynolds_number: float, 
                                  mach: float) -> Tuple[np.ndarray, float]:
        """
        Simple viscous-inviscid interaction using displacement thickness
        """
        # Simplified boundary layer calculation
        # Professional implementation would use integral methods or CFD coupling
        
        # Estimate boundary layer displacement thickness
        x_transition = 0.1  # Transition point (10% chord)
        rex_transition = reynolds_number * x_transition
        
        # Laminar and turbulent boundary layer correlations
        if rex_transition < 5e5:
            # Laminar
            delta_star_ratio = 1.72 / math.sqrt(rex_transition)
        else:
            # Turbulent
            delta_star_ratio = 0.046 / (rex_transition**0.2)
        
        # Modify pressure distribution due to displacement thickness
        # Simplified approach - uniform modification
        viscous_correction = delta_star_ratio * 0.1  # Scaling factor
        cp_viscous = inviscid_cp - viscous_correction
        
        # Estimate viscous drag increase
        cd_viscous_increase = delta_star_ratio * 0.002
        
        return cp_viscous, cd_viscous_increase

class AdvancedStructures:
    """Advanced structural analysis capabilities"""
    
    @staticmethod
    @handle_analysis_errors
    def tsai_wu_failure_criterion(stress_state: Dict[str, float], 
                                  material_properties: Dict[str, float]) -> CompositeFailureAnalysis:
        """
        Tsai-Wu failure criterion for composite materials
        Comprehensive failure analysis for aerospace composites
        """
        # Extract stress components (in principal material directions)
        sigma_11 = stress_state.get('sigma_11', 0)  # Fiber direction
        sigma_22 = stress_state.get('sigma_22', 0)  # Transverse direction
        tau_12 = stress_state.get('tau_12', 0)      # In-plane shear
        
        # Extract material strengths
        F_1t = material_properties.get('F_1t', 1500e6)  # Fiber tensile strength
        F_1c = material_properties.get('F_1c', 1200e6)  # Fiber compressive strength
        F_2t = material_properties.get('F_2t', 50e6)    # Transverse tensile strength
        F_2c = material_properties.get('F_2c', 200e6)   # Transverse compressive strength
        F_6 = material_properties.get('F_6', 70e6)      # In-plane shear strength
        
        # Tsai-Wu strength parameters
        F_1 = 1/F_1t - 1/F_1c
        F_2 = 1/F_2t - 1/F_2c
        F_11 = 1/(F_1t * F_1c)
        F_22 = 1/(F_2t * F_2c)
        F_66 = 1/F_6**2
        F_12 = -0.5 * math.sqrt(F_11 * F_22)  # Interaction term (simplified)
        
        # Tsai-Wu failure index
        tsai_wu_index = (F_1 * sigma_11 + F_2 * sigma_22 + 
                        F_11 * sigma_11**2 + F_22 * sigma_22**2 + 
                        F_66 * tau_12**2 + 2 * F_12 * sigma_11 * sigma_22)
        
        # Hashin failure criteria (separate fiber and matrix failure)
        # Fiber failure
        if sigma_11 >= 0:
            hashin_fiber = (sigma_11 / F_1t)**2 + (tau_12 / F_6)**2
        else:
            hashin_fiber = (sigma_11 / F_1c)**2
        
        # Matrix failure
        if sigma_22 >= 0:
            hashin_matrix = (sigma_22 / F_2t)**2 + (tau_12 / F_6)**2
        else:
            hashin_matrix = ((sigma_22 / (2 * F_6))**2 - 1) * (sigma_22 / F_2c) + (sigma_22 / F_2c)**2 + (tau_12 / F_6)**2
        
        # Determine critical failure mode
        failure_modes = {
            'Tsai-Wu': tsai_wu_index,
            'Hashin Fiber': hashin_fiber,
            'Hashin Matrix': hashin_matrix
        }
        
        critical_mode = max(failure_modes, key=failure_modes.get)
        max_index = failure_modes[critical_mode]
        
        # Calculate first ply failure load
        if max_index > 0:
            first_ply_failure_load = 1 / math.sqrt(max_index)
        else:
            first_ply_failure_load = float('inf')
        
        # Margin of safety
        margin_of_safety = first_ply_failure_load - 1.0
        
        return CompositeFailureAnalysis(
            tsai_wu_index=tsai_wu_index,
            hashin_fiber_failure=hashin_fiber,
            hashin_matrix_failure=hashin_matrix,
            first_ply_failure_load=first_ply_failure_load,
            failure_mode=critical_mode,
            margin_of_safety=margin_of_safety
        )
    
    @staticmethod
    @handle_analysis_errors
    def panel_buckling_analysis(panel_geometry: Dict[str, float], 
                               loading: Dict[str, float],
                               material_properties: Dict[str, float]) -> Dict[str, float]:
        """
        Advanced panel buckling analysis for aerospace structures
        Includes local and global buckling modes
        """
        # Panel dimensions
        a = panel_geometry['length']  # Length
        b = panel_geometry['width']   # Width
        t = panel_geometry['thickness']
        
        # Material properties
        E = material_properties['elastic_modulus']
        nu = material_properties['poisson_ratio']
        
        # Applied loading
        Nx = loading.get('Nx', 0)  # In-plane load in x-direction
        Ny = loading.get('Ny', 0)  # In-plane load in y-direction
        Nxy = loading.get('Nxy', 0)  # In-plane shear
        
        # Flexural rigidity
        D = E * t**3 / (12 * (1 - nu**2))
        
        # Buckling coefficients for different boundary conditions
        # Simply supported on all edges
        if abs(Nxy) < 1e-6:  # Pure compression
            if abs(Ny) < 1e-6:  # Uniaxial compression in x
                aspect_ratio = a / b
                if aspect_ratio >= 1:
                    k = 4.0 + (aspect_ratio)**2
                else:
                    k = 4.0 + (1/aspect_ratio)**2
            else:  # Biaxial compression
                # Simplified interaction formula
                k = 4.0  # Conservative estimate
        else:  # Shear buckling
            k_shear = 5.34 + 4.0 * (b/a)**2  # For simply supported
            k = k_shear
        
        # Critical buckling stress
        sigma_cr = k * math.pi**2 * D / (b**2 * t)
        
        # Critical buckling load
        if abs(Nx) > abs(Ny):
            N_cr = sigma_cr * t
            load_ratio = abs(Nx) / N_cr if N_cr > 0 else float('inf')
        else:
            N_cr = sigma_cr * t  # Simplified
            load_ratio = abs(Ny) / N_cr if N_cr > 0 else float('inf')
        
        # Buckling safety factor
        buckling_sf = 1 / load_ratio if load_ratio > 0 else float('inf')
        
        return {
            'critical_stress': sigma_cr,
            'critical_load': N_cr,
            'buckling_coefficient': k,
            'load_ratio': load_ratio,
            'buckling_safety_factor': buckling_sf,
            'buckling_mode': 'local' if min(a, b) / t > 50 else 'global'
        }
    
    @staticmethod
    @handle_analysis_errors
    def fatigue_life_analysis(stress_history: np.ndarray, 
                             material_properties: Dict[str, float],
                             mean_stress: float = 0.0) -> Dict[str, float]:
        """
        Fatigue life analysis using Miner's rule and SN curves
        """
        # Material fatigue properties
        Sut = material_properties.get('ultimate_tensile_strength', 500e6)
        Se = material_properties.get('endurance_limit', 0.5 * Sut)  # Estimated
        
        # SN curve parameters (simplified)
        # Professional implementation would use actual test data
        b = -0.1  # Fatigue strength exponent
        c = -0.5  # Fatigue ductility exponent
        
        # Rainflow counting (simplified implementation)
        # Professional code would use proper rainflow counting algorithm
        stress_ranges = np.diff(stress_history)
        stress_amplitudes = np.abs(stress_ranges) / 2
        
        # Count cycles at each amplitude level
        amplitude_bins = np.linspace(0, np.max(stress_amplitudes), 20)
        cycle_counts, _ = np.histogram(stress_amplitudes, bins=amplitude_bins)
        amplitude_centers = (amplitude_bins[:-1] + amplitude_bins[1:]) / 2
        
        # Calculate cycles to failure for each amplitude (SN curve)
        cycles_to_failure = []
        for amplitude in amplitude_centers:
            if amplitude < Se:
                # Infinite life region
                nf = float('inf')
            else:
                # Finite life region
                nf = (amplitude / Sut)**(-1/b) * 1000  # Simplified SN relation
            cycles_to_failure.append(nf)
        
        # Miner's rule - cumulative damage
        damage = 0.0
        for i, cycles_applied in enumerate(cycle_counts):
            if cycles_to_failure[i] < float('inf'):
                damage += cycles_applied / cycles_to_failure[i]
        
        # Fatigue life
        if damage > 0:
            fatigue_life = 1 / damage  # Life in terms of the given load spectrum
        else:
            fatigue_life = float('inf')
        
        # Safety factor for fatigue
        target_life = 1e6  # Target design life (cycles)
        fatigue_sf = fatigue_life / target_life if fatigue_life < float('inf') else float('inf')
        
        return {
            'fatigue_life_cycles': fatigue_life,
            'cumulative_damage': damage,
            'fatigue_safety_factor': fatigue_sf,
            'endurance_limit': Se,
            'critical_amplitude': np.max(stress_amplitudes),
            'analysis_method': 'Miners_Rule_SN_Curve'
        }
    
    @staticmethod
    @handle_analysis_errors
    def thermal_stress_analysis(temperature_field: np.ndarray,
                               material_properties: Dict[str, float],
                               constraint_conditions: Dict[str, Any]) -> Dict[str, float]:
        """
        Thermal stress analysis for aerospace structures
        """
        # Material thermal properties
        E = material_properties['elastic_modulus']
        nu = material_properties['poisson_ratio']
        alpha = material_properties.get('thermal_expansion', 12e-6)  # /°C
        
        # Temperature difference from reference
        T_ref = constraint_conditions.get('reference_temperature', 20)  # °C
        delta_T = np.mean(temperature_field) - T_ref
        
        # Thermal strain
        epsilon_thermal = alpha * delta_T
        
        # Thermal stress (constrained expansion)
        if constraint_conditions.get('fully_constrained', False):
            # Fully constrained - maximum thermal stress
            sigma_thermal = E * epsilon_thermal / (1 - nu)
        else:
            # Partially constrained - reduced stress
            constraint_factor = constraint_conditions.get('constraint_factor', 0.5)
            sigma_thermal = E * epsilon_thermal * constraint_factor / (1 - nu)
        
        # Thermal stress safety factor
        allowable_stress = material_properties.get('yield_strength', 250e6)
        thermal_sf = allowable_stress / abs(sigma_thermal) if abs(sigma_thermal) > 0 else float('inf')
        
        return {
            'thermal_stress': sigma_thermal,
            'thermal_strain': epsilon_thermal,
            'temperature_change': delta_T,
            'thermal_safety_factor': thermal_sf,
            'max_temperature': np.max(temperature_field),
            'min_temperature': np.min(temperature_field)
        }

class AeroelasticCoupling:
    """Simple aeroelastic coupling for flexible structures"""
    
    @staticmethod
    @handle_analysis_errors
    def divergence_analysis(wing_properties: Dict[str, float],
                           flight_conditions: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate divergence speed for wing structures
        """
        # Wing structural properties
        GJ = wing_properties.get('torsional_rigidity', 1e6)  # Nm²
        chord = wing_properties.get('chord', 2.0)  # m
        span = wing_properties.get('span', 10.0)  # m
        
        # Aerodynamic center offset from elastic axis
        x_ac = wing_properties.get('aerodynamic_center_offset', 0.1)  # % chord
        
        # Flight conditions
        rho = flight_conditions.get('air_density', 1.225)  # kg/m³
        
        # Simplified divergence calculation
        # Professional analysis would use detailed structural and aerodynamic models
        
        # Aerodynamic lift curve slope (per radian)
        cl_alpha = 2 * math.pi  # Simplified
        
        # Divergence dynamic pressure
        q_div = GJ / (chord * span * x_ac * cl_alpha)
        
        # Divergence speed
        V_div = math.sqrt(2 * q_div / rho)
        
        return {
            'divergence_speed': V_div,
            'divergence_dynamic_pressure': q_div,
            'torsional_rigidity': GJ,
            'aerodynamic_center_offset': x_ac
        }

# Advanced analysis factory
class AdvancedAnalysisFactory:
    """Factory for advanced analysis capabilities"""
    
    def __init__(self):
        self.aerodynamics = AdvancedAerodynamics()
        self.structures = AdvancedStructures()
        self.aeroelastics = AeroelasticCoupling()
    
    def get_compressibility_correction(self, method: str = "karman_tsien"):
        """Get appropriate compressibility correction method"""
        if method.lower() == "prandtl_glauert":
            return self.aerodynamics.prandtl_glauert_correction
        elif method.lower() == "karman_tsien":
            return self.aerodynamics.karman_tsien_correction
        else:
            raise ValueError(f"Unknown compressibility correction method: {method}")
    
    def get_failure_analysis(self, material_type: str = "composite"):
        """Get appropriate failure analysis method"""
        if material_type.lower() == "composite":
            return self.structures.tsai_wu_failure_criterion
        else:
            raise ValueError(f"Failure analysis not implemented for: {material_type}")

# Global advanced analysis instance
advanced_analysis = AdvancedAnalysisFactory()