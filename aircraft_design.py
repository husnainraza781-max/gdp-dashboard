"""
Aircraft Design Module - Raymer-Based Conceptual Design
Implements complete aircraft design workflow from Raymer's Aircraft Design: A Conceptual Approach
"""

import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional, Tuple, Any
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import differential_evolution, minimize
from dataclasses import dataclass, field, asdict
import io
import base64
from pathlib import Path
import math

def sanitize_complex_values(value):
    """
    Sanitize values that might be complex numbers or NaN
    """
    if isinstance(value, complex):
        return float(value.real)
    elif isinstance(value, (np.number, np.ndarray)):
        if np.iscomplexobj(value):
            return float(np.real(value))
        elif np.isnan(value) or np.isinf(value):
            return 0.0
        else:
            return float(value)
    elif isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return float(value)
    else:
        return 0.0


def sanitize_design(design: 'AircraftDesign') -> 'AircraftDesign':
    """
    Sanitize all numerical values in aircraft design to ensure they are real numbers
    """
    # List of all numerical attributes that need sanitization
    numerical_attrs = [
        'mtow', 'oew', 'fuel_weight', 'payload_weight',
        'wing_area', 'wingspan', 'aspect_ratio', 'taper_ratio', 'sweep_angle',
        'root_chord', 'tip_chord', 'mac',
        'h_tail_area', 'v_tail_area', 'tail_arm',
        'fuselage_length', 'fuselage_diameter',
        'cl_cruise', 'cl_max', 'cd0', 'oswald_e', 'ld_max', 'ld_cruise',
        'thrust_required', 'engine_count', 'sfc',
        'range_km', 'endurance_hrs', 'cruise_speed', 'stall_speed',
        'climb_rate', 'ceiling', 'takeoff_distance',
        'static_margin'
    ]
    
    for attr in numerical_attrs:
        if hasattr(design, attr):
            value = getattr(design, attr)
            sanitized_value = sanitize_complex_values(value)
            setattr(design, attr, sanitized_value)
    
    # Handle tuple attributes like cg_range
    if hasattr(design, 'cg_range') and design.cg_range:
        design.cg_range = (
            sanitize_complex_values(design.cg_range[0]),
            sanitize_complex_values(design.cg_range[1])
        )
    
    return design

# Aircraft type configurations and defaults
AIRCRAFT_CONFIGS = {
    "Fighter": {
        "cruise_speed": 450,  # kts
        "cruise_alt": 35000,  # ft
        "range": 1500,  # km
        "payload": 2000,  # kg
        "crew": 1,
        "wing_loading": 400,  # kg/m²
        "thrust_weight": 1.2,
        "aspect_ratio": 3.5,
        "sweep": 35,  # degrees
        "taper": 0.2,
        "t_c": 0.06,  # thickness to chord
        "bypass_ratio": 0.4
    },
    "Passenger": {
        "cruise_speed": 460,  # kts
        "cruise_alt": 39000,  # ft
        "range": 5000,  # km
        "passengers": 150,
        "crew": 5,
        "wing_loading": 600,  # kg/m²
        "thrust_weight": 0.3,
        "aspect_ratio": 9.5,
        "sweep": 25,  # degrees
        "taper": 0.3,
        "t_c": 0.12,
        "bypass_ratio": 5.0
    },
    "UAV": {
        "cruise_speed": 100,  # kts
        "cruise_alt": 20000,  # ft
        "endurance": 24,  # hours
        "payload": 50,  # kg
        "crew": 0,
        "wing_loading": 50,  # kg/m²
        "thrust_weight": 0.2,
        "aspect_ratio": 15,
        "sweep": 0,  # degrees
        "taper": 0.5,
        "t_c": 0.15,
        "electric": True
    },
    "Other": {
        "cruise_speed": 200,  # kts
        "cruise_alt": 10000,  # ft
        "range": 1000,  # km
        "payload": 500,  # kg
        "crew": 2,
        "wing_loading": 300,  # kg/m²
        "thrust_weight": 0.4,
        "aspect_ratio": 7.0,
        "sweep": 5,  # degrees
        "taper": 0.5,
        "t_c": 0.12
    }
}

@dataclass
class AircraftDesign:
    """Complete aircraft design data structure"""
    # Basic configuration
    aircraft_type: str
    mission_profile: Dict[str, Any]
    
    # Weights
    mtow: float = 0.0  # kg
    oew: float = 0.0  # kg
    fuel_weight: float = 0.0
    payload_weight: float = 0.0
    
    # Wing geometry
    wing_area: float = 0.0  # m²
    wingspan: float = 0.0  # m
    aspect_ratio: float = 7.0
    taper_ratio: float = 0.5
    sweep_angle: float = 0.0  # degrees
    root_chord: float = 0.0
    tip_chord: float = 0.0
    mac: float = 0.0  # mean aerodynamic chord
    
    # Tail geometry
    h_tail_area: float = 0.0
    v_tail_area: float = 0.0
    tail_arm: float = 0.0
    
    # Fuselage
    fuselage_length: float = 0.0
    fuselage_diameter: float = 0.0
    
    # Aerodynamics
    cl_cruise: float = 0.0
    cl_max: float = 0.0
    cd0: float = 0.0
    oswald_e: float = 0.8
    ld_max: float = 0.0
    ld_cruise: float = 0.0
    
    # Propulsion
    thrust_required: float = 0.0  # N
    engine_count: int = 1
    sfc: float = 0.0  # kg/N/s
    
    # Performance
    range_km: float = 0.0
    endurance_hrs: float = 0.0
    cruise_speed: float = 0.0  # m/s
    stall_speed: float = 0.0
    climb_rate: float = 0.0  # m/s
    ceiling: float = 0.0  # m
    takeoff_distance: float = 0.0  # m
    
    # Stability
    static_margin: float = 0.0
    cg_range: Tuple[float, float] = (0.0, 0.0)
    
    # Iteration data
    iterations: List[Dict] = field(default_factory=list)
    converged: bool = False
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return asdict(self)


class RaymerDesignEngine:
    """
    Implements Raymer's aircraft design methodology
    Based on Aircraft Design: A Conceptual Approach by Daniel P. Raymer
    """
    
    def __init__(self):
        """Initialize design engine with Raymer constants"""
        # Statistical weight coefficients from Raymer
        self.weight_coeffs = {
            "Fighter": {
                "A": -0.02,
                "B": 1.04,
                "C1": 0.91,  # Empty weight fraction coefficients
                "fuel_frac_cruise": 0.85,
                "fuel_frac_loiter": 0.99
            },
            "Passenger": {
                "A": -0.10,
                "B": 1.06,
                "C1": 0.96,
                "fuel_frac_cruise": 0.94,
                "fuel_frac_loiter": 0.985
            },
            "UAV": {
                "A": 0.04,
                "B": 0.98,
                "C1": 0.89,
                "fuel_frac_cruise": 0.98,  # Electric or high efficiency
                "fuel_frac_loiter": 0.995
            },
            "Other": {
                "A": -0.05,
                "B": 1.02,
                "C1": 0.93,
                "fuel_frac_cruise": 0.90,
                "fuel_frac_loiter": 0.99
            }
        }
        
        # Aerodynamic constants from Raymer
        self.aero_constants = {
            "cl_max_clean": {
                "Fighter": 1.2,
                "Passenger": 1.4,
                "UAV": 1.6,
                "Other": 1.5
            },
            "cl_max_landing": {
                "Fighter": 1.8,
                "Passenger": 2.4,
                "UAV": 2.0,
                "Other": 2.2
            }
        }
    
    def design_aircraft(self, aircraft_type: str, mission_req: Dict, 
                        progress_callback=None) -> AircraftDesign:
        """
        Main design function implementing Raymer's design process
        
        Args:
            aircraft_type: Fighter, Passenger, UAV, or Other
            mission_req: Mission requirements dictionary
            progress_callback: Optional callback for progress updates
            
        Returns:
            Complete AircraftDesign object
        """
        design = AircraftDesign(
            aircraft_type=aircraft_type,
            mission_profile=mission_req
        )
        
        # Step 1: Initial sizing
        if progress_callback:
            progress_callback("Step 1: Initial weight and geometry sizing...")
        design = self._initial_sizing(design)
        
        # Step 2: Weight estimation
        if progress_callback:
            progress_callback("Step 2: Detailed weight estimation...")
        design = self._weight_estimation(design)
        
        # Step 3: Wing design
        if progress_callback:
            progress_callback("Step 3: Wing geometry and planform design...")
        design = self._wing_design(design)
        
        # Step 4: Fuselage design
        if progress_callback:
            progress_callback("Step 4: Fuselage sizing...")
        design = self._fuselage_design(design)
        
        # Step 5: Tail design
        if progress_callback:
            progress_callback("Step 5: Empennage sizing...")
        design = self._tail_design(design)
        
        # Step 6: Aerodynamic analysis
        if progress_callback:
            progress_callback("Step 6: Aerodynamic analysis...")
        design = self._aerodynamic_analysis(design)
        
        # Step 7: Propulsion sizing
        if progress_callback:
            progress_callback("Step 7: Propulsion system sizing...")
        design = self._propulsion_sizing(design)
        
        # Step 8: Performance calculation
        if progress_callback:
            progress_callback("Step 8: Performance calculations...")
        design = self._performance_calc(design)
        
        # Step 9: Stability analysis
        if progress_callback:
            progress_callback("Step 9: Stability and control analysis...")
        design = self._stability_analysis(design)
        
        # Step 10: Iteration and convergence
        if progress_callback:
            progress_callback("Step 10: Design iteration and convergence...")
        design = self._iterate_design(design, progress_callback)
        
        # Final sanitization to ensure all values are real numbers
        design = sanitize_design(design)
        
        return design
    
    def _initial_sizing(self, design: AircraftDesign) -> AircraftDesign:
        """Initial sizing using Raymer's statistical methods"""
        config = AIRCRAFT_CONFIGS[design.aircraft_type]
        mission = design.mission_profile
        
        # Estimate initial MTOW using Raymer's method
        # W0 = Wpayload / (1 - Wfuel/W0 - Wempty/W0)
        
        # Initial guess based on payload
        payload = mission.get('payload', config.get('payload', 1000))
        
        # Raymer's fuel fraction estimation (simplified)
        range_km = mission.get('range', config.get('range', 1000))
        endurance_hrs = mission.get('endurance', 0)
        
        # Mission fuel fraction (Raymer Ch. 3) - ensure reasonable bounds
        if endurance_hrs > 0:
            fuel_fraction = max(0.15 + 0.02 * min(endurance_hrs, 20), 0.1)  # Cap endurance, min 10%
        else:
            fuel_fraction = max(0.06 + 0.00012 * min(range_km, 20000), 0.1)  # Cap range, min 10%
        
        # Clamp fuel fraction to reasonable bounds
        fuel_fraction = min(max(fuel_fraction, 0.1), 0.6)  # Between 10% and 60%
        
        # Empty weight fraction (Raymer statistical) - ensure positive result
        coeffs = self.weight_coeffs[design.aircraft_type]
        We_W0_raw = coeffs['A'] + coeffs['B'] * (1 / (1 + fuel_fraction))
        
        # Ensure empty weight fraction is within realistic bounds
        if design.aircraft_type == "Fighter":
            We_W0 = max(min(We_W0_raw, 0.65), 0.45)  # 45-65% for fighters
        elif design.aircraft_type == "Passenger":
            We_W0 = max(min(We_W0_raw, 0.60), 0.40)  # 40-60% for passenger
        else:
            We_W0 = max(min(We_W0_raw, 0.70), 0.35)  # 35-70% for others
        
        # Ensure the denominator is positive for MTOW calculation
        denominator = 1 - fuel_fraction - We_W0
        if denominator <= 0.05:  # If less than 5%, use fallback calculation
            # Fallback to simple weight estimation
            if design.aircraft_type == "Fighter":
                design.mtow = payload * 4.0  # Typical fighter multiplier
            elif design.aircraft_type == "Passenger":
                design.mtow = payload * 2.5  # Typical passenger multiplier
            else:
                design.mtow = payload * 3.0  # Default multiplier
        else:
            design.mtow = payload / denominator
        
        # Ensure MTOW is reasonable
        design.mtow = max(design.mtow, payload * 1.5)  # At least 1.5x payload
        design.mtow = min(design.mtow, payload * 20)   # At most 20x payload
        
        design.payload_weight = payload
        design.fuel_weight = design.mtow * fuel_fraction
        design.oew = design.mtow * We_W0
        
        return design
    
    def _weight_estimation(self, design: AircraftDesign) -> AircraftDesign:
        """Detailed weight estimation using Raymer's component buildup"""
        # Component weight fractions (Raymer Ch. 15)
        
        # Wing weight (Raymer equation)
        Nz = 3.5  # Ultimate load factor
        tc = 0.12  # Thickness to chord ratio
        lambda_sweep = np.radians(design.sweep_angle)
        
        # Simplified Raymer wing weight equation
        W_wing = 0.036 * (design.mtow * Nz)**0.758 * design.wing_area**0.6 * \
                design.aspect_ratio**0.4 * (100 * tc)**(-0.3) * \
                (1 + design.taper_ratio)**0.04 * np.cos(lambda_sweep)**(-1.0)
        
        # Fuselage weight
        W_fuselage = 0.052 * design.fuselage_length**1.086 * \
                    (design.mtow * Nz)**0.177
        
        # Tail weight (horizontal + vertical)
        W_tail = 0.016 * (design.mtow * Nz)**0.414 * \
                (design.h_tail_area + design.v_tail_area)**1.0
        
        # Landing gear weight
        W_gear = 0.04 * design.mtow
        
        # Systems and equipment
        W_systems = 0.10 * design.mtow  # Simplified
        
        # Propulsion weight (installed)
        if design.aircraft_type == "Fighter":
            W_prop = 0.15 * design.mtow
        elif design.aircraft_type == "Passenger":
            W_prop = 0.12 * design.mtow
        else:
            W_prop = 0.08 * design.mtow
        
        # Update OEW
        design.oew = W_wing + W_fuselage + W_tail + W_gear + W_systems + W_prop
        
        # Recalculate fuel weight
        design.fuel_weight = design.mtow - design.oew - design.payload_weight
        
        return design
    
    def _wing_design(self, design: AircraftDesign) -> AircraftDesign:
        """Wing planform design using Raymer methods"""
        config = AIRCRAFT_CONFIGS[design.aircraft_type]
        
        # Wing loading from config or mission requirements
        W_S = config['wing_loading']  # kg/m²
        design.wing_area = design.mtow / W_S
        
        # Aspect ratio
        design.aspect_ratio = config['aspect_ratio']
        design.wingspan = np.sqrt(design.wing_area * design.aspect_ratio)
        
        # Planform parameters
        design.taper_ratio = config['taper']
        design.sweep_angle = config['sweep']
        
        # Calculate chord distribution
        design.root_chord = 2 * design.wing_area / \
                           (design.wingspan * (1 + design.taper_ratio))
        design.tip_chord = design.root_chord * design.taper_ratio
        
        # Mean aerodynamic chord (MAC)
        design.mac = (2/3) * design.root_chord * \
                    (1 + design.taper_ratio + design.taper_ratio**2) / \
                    (1 + design.taper_ratio)
        
        return design
    
    def _fuselage_design(self, design: AircraftDesign) -> AircraftDesign:
        """Fuselage sizing using Raymer guidelines"""
        if design.aircraft_type == "Fighter":
            # Fighter: slender fuselage
            design.fuselage_length = 15.0 * design.mtow**0.25
            design.fuselage_diameter = design.fuselage_length / 10
        elif design.aircraft_type == "Passenger":
            # Passenger: based on seating
            passengers = design.mission_profile.get('passengers', 150)
            seats_per_row = 6  # Typical narrow-body
            row_pitch = 0.8  # meters
            num_rows = np.ceil(passengers / seats_per_row)
            design.fuselage_length = 5 + num_rows * row_pitch + 10  # Cockpit + cabin + tail
            design.fuselage_diameter = 3.5 + 0.01 * passengers  # Typical for capacity
        else:
            # General/UAV
            design.fuselage_length = 7.0 * design.mtow**0.3
            design.fuselage_diameter = design.fuselage_length / 8
        
        return design
    
    def _tail_design(self, design: AircraftDesign) -> AircraftDesign:
        """Tail sizing using volume coefficient method (Raymer Ch. 6)"""
        # Tail volume coefficients (typical values from Raymer)
        if design.aircraft_type == "Fighter":
            c_ht = 0.4  # Horizontal tail volume coefficient
            c_vt = 0.04  # Vertical tail volume coefficient
        elif design.aircraft_type == "Passenger":
            c_ht = 0.9
            c_vt = 0.09
        else:
            c_ht = 0.6
            c_vt = 0.06
        
        # Tail arm (distance from wing MAC to tail MAC)
        design.tail_arm = 0.5 * design.fuselage_length
        
        # Horizontal tail area
        design.h_tail_area = (c_ht * design.mac * design.wing_area) / design.tail_arm
        
        # Vertical tail area
        design.v_tail_area = (c_vt * design.wingspan * design.wing_area) / design.tail_arm
        
        return design
    
    def _aerodynamic_analysis(self, design: AircraftDesign) -> AircraftDesign:
        """Aerodynamic performance estimation using Raymer methods"""
        # CL max estimation
        design.cl_max = self.aero_constants["cl_max_clean"][design.aircraft_type]
        
        # Cruise CL
        cruise_alt = design.mission_profile.get('cruise_alt', 30000) * 0.3048  # Convert to meters
        rho = self._get_density(cruise_alt)
        V_cruise = design.mission_profile.get('cruise_speed', 200) * 0.514444  # kts to m/s
        design.cruise_speed = V_cruise
        
        # Calculate cruise CL
        design.cl_cruise = (2 * design.mtow * 9.81) / (rho * V_cruise**2 * design.wing_area)
        
        # Oswald efficiency factor (Raymer Fig 12.14)
        design.oswald_e = 0.85 - 0.05 * design.sweep_angle / 30  # Simplified
        
        # Zero-lift drag coefficient (Raymer method)
        # Based on wetted area and skin friction
        S_wet_S_ref = 3.5 + 0.5 * design.aspect_ratio  # Wetted area ratio estimate
        Cf = 0.003  # Skin friction coefficient
        design.cd0 = S_wet_S_ref * Cf
        
        # Induced drag coefficient
        cdi_cruise = design.cl_cruise**2 / (np.pi * design.aspect_ratio * design.oswald_e)
        
        # Total drag
        cd_cruise = design.cd0 + cdi_cruise
        
        # L/D ratios
        design.ld_cruise = design.cl_cruise / cd_cruise
        design.ld_max = np.sqrt(np.pi * design.aspect_ratio * design.oswald_e / (4 * design.cd0))
        
        # Stall speed
        design.stall_speed = np.sqrt((2 * design.mtow * 9.81) / 
                                    (1.225 * design.cl_max * design.wing_area))
        
        return design
    
    def _propulsion_sizing(self, design: AircraftDesign) -> AircraftDesign:
        """Propulsion system sizing using Raymer methods"""
        config = AIRCRAFT_CONFIGS[design.aircraft_type]
        
        # Thrust to weight ratio
        T_W = config['thrust_weight']
        design.thrust_required = T_W * design.mtow * 9.81  # N
        
        # Number of engines
        if design.aircraft_type == "Fighter":
            design.engine_count = 1 if design.thrust_required < 100000 else 2
        elif design.aircraft_type == "Passenger":
            design.engine_count = 2 if design.mtow < 100000 else 4
        else:
            design.engine_count = 1
        
        # Specific fuel consumption (Raymer typical values)
        if design.aircraft_type == "Fighter":
            design.sfc = 0.8 / 3600  # lb/lbf/hr to kg/N/s
        elif design.aircraft_type == "Passenger":
            design.sfc = 0.5 / 3600
        elif design.aircraft_type == "UAV" and config.get('electric'):
            design.sfc = 0.0  # Electric
        else:
            design.sfc = 0.6 / 3600
        
        return design
    
    def _performance_calc(self, design: AircraftDesign) -> AircraftDesign:
        """Performance calculations using Raymer methods"""
        # Range calculation (Breguet equation for jet aircraft)
        if design.sfc > 0:
            design.range_km = (design.cruise_speed / design.sfc / 9.81) * \
                             design.ld_cruise * \
                             np.log((design.mtow) / 
                                   (design.mtow - design.fuel_weight)) / 1000
        else:
            # Electric aircraft - battery limited
            design.range_km = design.mission_profile.get('range', 100)
        
        # Endurance calculation
        if design.aircraft_type == "UAV":
            # Endurance more important for UAV
            design.endurance_hrs = design.fuel_weight / \
                                  (design.sfc * design.thrust_required / design.ld_cruise) / 3600
        else:
            design.endurance_hrs = design.range_km / (design.cruise_speed * 3.6)
        
        # Climb rate (simplified)
        excess_power = (design.thrust_required * design.cruise_speed - 
                       design.mtow * 9.81 * design.cruise_speed / design.ld_cruise)
        design.climb_rate = excess_power / (design.mtow * 9.81)
        
        # Service ceiling (simplified - where climb rate = 100 ft/min = 0.5 m/s)
        design.ceiling = 10000 + 1000 * design.climb_rate  # Simplified estimate
        
        # Takeoff distance (Raymer simplified method)
        # Based on wing loading and CL_max
        V_liftoff = 1.2 * design.stall_speed
        a_avg = design.thrust_required / design.mtow - 0.02 * 9.81  # Average acceleration
        design.takeoff_distance = V_liftoff**2 / (2 * a_avg)
        
        return design
    
    def _stability_analysis(self, design: AircraftDesign) -> AircraftDesign:
        """Stability and control analysis using Raymer methods"""
        # Static margin (simplified)
        # SM = (x_np - x_cg) / MAC
        
        # Neutral point estimate (% MAC)
        x_np = 0.25 + 0.4 * (design.h_tail_area / design.wing_area) * \
               (design.tail_arm / design.mac)
        
        # CG range estimate
        x_cg_fwd = 0.20  # 20% MAC
        x_cg_aft = 0.35  # 35% MAC
        
        design.static_margin = x_np - x_cg_aft
        design.cg_range = (x_cg_fwd, x_cg_aft)
        
        return design
    
    def _iterate_design(self, design: AircraftDesign, progress_callback=None) -> AircraftDesign:
        """Iterate design until convergence"""
        max_iterations = 10
        tolerance = 0.01  # 1% convergence criteria
        
        for iteration in range(max_iterations):
            if progress_callback:
                progress_callback(f"Iteration {iteration + 1}/{max_iterations}...")
            
            # Store current values
            mtow_old = design.mtow
            
            # Recalculate weights
            design = self._weight_estimation(design)
            
            # Check convergence
            delta = abs(design.mtow - mtow_old) / mtow_old
            
            # Store iteration data
            design.iterations.append({
                "iteration": iteration + 1,
                "mtow": design.mtow,
                "delta": delta
            })
            
            if delta < tolerance:
                design.converged = True
                if progress_callback:
                    progress_callback(f"Converged after {iteration + 1} iterations!")
                break
            
            # Update dependent parameters
            design = self._wing_design(design)
            design = self._aerodynamic_analysis(design)
        
        return design
    
    def _get_density(self, altitude_m: float) -> float:
        """Atmospheric density using standard atmosphere"""
        if altitude_m < 11000:
            T = 288.15 - 0.0065 * altitude_m
            p = 101325 * (T / 288.15)**5.256
        else:
            T = 216.65
            p = 22632.1 * np.exp(-0.0001577 * (altitude_m - 11000))
        
        rho = p / (287.05 * T)
        return rho
    
    def optimize_design(self, design: AircraftDesign, objective: str) -> AircraftDesign:
        """
        Optimize design for specific objective
        
        Args:
            design: Initial design
            objective: 'range', 'endurance', 'payload', 'weight'
        """
        def objective_func(x):
            # x = [wing_area, aspect_ratio, sweep, taper]
            temp_design = design
            temp_design.wing_area = x[0]
            temp_design.aspect_ratio = x[1]
            temp_design.sweep_angle = x[2]
            temp_design.taper_ratio = x[3]
            
            # Recalculate performance
            temp_design = self._wing_design(temp_design)
            temp_design = self._aerodynamic_analysis(temp_design)
            temp_design = self._performance_calc(temp_design)
            
            # Return negative for maximization
            if objective == 'range':
                return -temp_design.range_km
            elif objective == 'endurance':
                return -temp_design.endurance_hrs
            elif objective == 'payload':
                return -temp_design.payload_weight
            elif objective == 'weight':
                return temp_design.oew
            else:
                return -temp_design.ld_cruise
        
        # Bounds for design variables - validate and ensure proper bounds
        wing_area_base = max(design.wing_area, 10.0)  # Ensure minimum valid wing area
        aspect_ratio_base = max(design.aspect_ratio, 3.0)  # Ensure minimum valid AR
        sweep_base = max(min(design.sweep_angle, 45.0), 0.0)  # Clamp sweep angle
        taper_base = max(min(design.taper_ratio, 1.0), 0.1)  # Clamp taper ratio
        
        bounds = [
            (wing_area_base * 0.5, wing_area_base * 1.5),  # Wing area
            (3.0, 20.0),  # Aspect ratio
            (0.0, 45.0),  # Sweep angle
            (0.1, 1.0)    # Taper ratio
        ]
        
        # Validate bounds - ensure lower < upper for all bounds
        validated_bounds = []
        for i, (lower, upper) in enumerate(bounds):
            if lower >= upper:
                # Fix invalid bounds with reasonable defaults
                if i == 0:  # Wing area
                    validated_bounds.append((10.0, 50.0))
                elif i == 1:  # Aspect ratio
                    validated_bounds.append((3.0, 20.0))
                elif i == 2:  # Sweep angle
                    validated_bounds.append((0.0, 45.0))
                elif i == 3:  # Taper ratio
                    validated_bounds.append((0.1, 1.0))
            else:
                validated_bounds.append((lower, upper))
        
        bounds = validated_bounds
        
        # Run optimization
        result = differential_evolution(objective_func, bounds, maxiter=50)
        
        # Apply optimized values
        design.wing_area = result.x[0]
        design.aspect_ratio = result.x[1]
        design.sweep_angle = result.x[2]
        design.taper_ratio = result.x[3]
        
        # Recalculate everything with optimized values
        design = self._wing_design(design)
        design = self._aerodynamic_analysis(design)
        design = self._performance_calc(design)
        
        return design


def create_3d_model(design: AircraftDesign) -> go.Figure:
    """
    Create realistic 3D parametric model of aircraft with professional geometry and surfaces
    """
    # Sanitize design data to prevent complex numbers and NaN values
    design = sanitize_design(design)
    
    # Validate and ensure realistic proportions based on aircraft type
    _validate_aircraft_proportions(design)
    
    fig = go.Figure()
    
    # Create realistic wing geometry with proper airfoil sections
    _add_realistic_wings(fig, design)
    
    # Create realistic fuselage with proper lofting
    _add_realistic_fuselage(fig, design)
    
    # Create realistic empennage (tail surfaces)
    _add_realistic_empennage(fig, design)
    
    # Add engine nacelles if needed
    if design.engine_count > 1:
        _add_realistic_engines(fig, design)
    
    # Configure professional lighting and camera
    _configure_professional_layout(fig, design)
    
    return fig


def _validate_aircraft_proportions(design: AircraftDesign):
    """Validate and enforce realistic aircraft proportions"""
    # Wing geometry validation based on aircraft type
    if design.aircraft_type == "Fighter":
        # Fighter aircraft proportions
        design.wingspan = max(design.wingspan, 8.0) if design.wingspan > 0 else 12.0
        design.fuselage_length = max(design.fuselage_length, 12.0) if design.fuselage_length > 0 else 16.0
        design.fuselage_diameter = max(design.fuselage_diameter, 1.2) if design.fuselage_diameter > 0 else 1.8
        design.root_chord = max(design.root_chord, 2.0) if design.root_chord > 0 else 4.0
        design.taper_ratio = max(min(design.taper_ratio, 0.4), 0.2) if design.taper_ratio > 0 else 0.3
        design.sweep_angle = max(min(design.sweep_angle, 45.0), 25.0) if design.sweep_angle > 0 else 35.0
    elif design.aircraft_type == "Passenger":
        # Passenger aircraft proportions
        design.wingspan = max(design.wingspan, 25.0) if design.wingspan > 0 else 35.0
        design.fuselage_length = max(design.fuselage_length, 25.0) if design.fuselage_length > 0 else 40.0
        design.fuselage_diameter = max(design.fuselage_diameter, 3.0) if design.fuselage_diameter > 0 else 4.0
        design.root_chord = max(design.root_chord, 4.0) if design.root_chord > 0 else 6.0
        design.taper_ratio = max(min(design.taper_ratio, 0.35), 0.25) if design.taper_ratio > 0 else 0.3
        design.sweep_angle = max(min(design.sweep_angle, 35.0), 20.0) if design.sweep_angle > 0 else 25.0
    else:
        # Default proportions
        design.wingspan = max(design.wingspan, 10.0) if design.wingspan > 0 else 15.0
        design.fuselage_length = max(design.fuselage_length, 8.0) if design.fuselage_length > 0 else 12.0
        design.fuselage_diameter = max(design.fuselage_diameter, 1.0) if design.fuselage_diameter > 0 else 1.5
        design.root_chord = max(design.root_chord, 1.5) if design.root_chord > 0 else 2.5
        design.taper_ratio = max(min(design.taper_ratio, 0.5), 0.3) if design.taper_ratio > 0 else 0.4
        design.sweep_angle = max(min(design.sweep_angle, 30.0), 0.0) if design.sweep_angle >= 0 else 15.0
    
    # Calculate tip chord from taper ratio
    design.tip_chord = design.root_chord * design.taper_ratio
    
    # Realistic tail sizing
    design.h_tail_area = design.wing_area * 0.25 if design.wing_area > 0 else design.wingspan * design.root_chord * 0.2
    design.v_tail_area = design.wing_area * 0.18 if design.wing_area > 0 else design.wingspan * design.root_chord * 0.15
    design.tail_arm = design.fuselage_length * 0.45
    


def _add_realistic_wings(fig: go.Figure, design: AircraftDesign):
    """Add realistic wing surfaces with proper airfoil geometry"""
    # High-resolution wing geometry
    n_span = 25  # Increased for smooth surfaces
    n_chord = 50  # High chord resolution for realistic airfoil shape
    
    span_stations = np.linspace(0, 1, n_span)
    chord_stations = np.linspace(0, 1, n_chord)
    
    # Realistic NACA airfoil thickness distribution based on aircraft type
    if design.aircraft_type == "Fighter":
        thickness_ratio = 0.06  # 6% thickness for fighters
        camber = 0.02  # 2% camber
    elif design.aircraft_type == "Passenger":
        thickness_ratio = 0.12  # 12% thickness for passenger aircraft
        camber = 0.04  # 4% camber
    else:
        thickness_ratio = 0.09  # 9% thickness for general aviation
        camber = 0.03  # 3% camber
    
    # Enhanced NACA 4-digit airfoil with camber
    y_thickness = 5 * thickness_ratio * (0.2969 * np.sqrt(chord_stations) - 
                                         0.1260 * chord_stations - 
                                         0.3516 * chord_stations**2 + 
                                         0.2843 * chord_stations**3 - 
                                         0.1015 * chord_stations**4)
    
    # Add camber line for realistic airfoil shape
    y_camber = np.where(chord_stations <= 0.4,
                       camber / 0.16 * (2 * 0.4 * chord_stations - chord_stations**2),
                       camber / 0.36 * (1 - 2 * 0.4 + 2 * 0.4 * chord_stations - chord_stations**2))
    
    # Create wing surface grids with twist and dihedral
    wing_x_upper = np.zeros((n_span, n_chord))
    wing_y_upper = np.zeros((n_span, n_chord))
    wing_z_upper = np.zeros((n_span, n_chord))
    wing_x_lower = np.zeros((n_span, n_chord))
    wing_y_lower = np.zeros((n_span, n_chord))
    wing_z_lower = np.zeros((n_span, n_chord))
    
    # Add realistic dihedral angle
    dihedral_angle = 3.0 if design.aircraft_type == "Passenger" else 1.0  # degrees
    washout_angle = -2.0  # degrees of washout at tip
    
    for i, span_frac in enumerate(span_stations):
        # Linear taper
        chord_length = design.root_chord * (1 - span_frac * (1 - design.taper_ratio))
        
        # Leading edge sweep
        y_le = -span_frac * design.wingspan/2 * np.tan(np.radians(design.sweep_angle))
        
        # Twist distribution (washout)
        twist = washout_angle * span_frac
        twist_rad = np.radians(twist)
        
        # Dihedral effect
        z_dihedral = span_frac * design.wingspan/2 * np.tan(np.radians(dihedral_angle))
        
        for j, chord_frac in enumerate(chord_stations):
            x_pos = span_frac * design.wingspan/2
            
            # Apply airfoil coordinates with twist
            upper_z = (y_thickness[j] + y_camber[j]) * chord_length
            lower_z = (-y_thickness[j] + y_camber[j]) * chord_length
            
            # Apply twist transformation
            y_local = y_le - chord_frac * chord_length
            y_twisted = y_local * np.cos(twist_rad) + upper_z * np.sin(twist_rad)
            z_upper_twisted = -y_local * np.sin(twist_rad) + upper_z * np.cos(twist_rad) + z_dihedral
            
            y_twisted_lower = y_local * np.cos(twist_rad) + lower_z * np.sin(twist_rad)
            z_lower_twisted = -y_local * np.sin(twist_rad) + lower_z * np.cos(twist_rad) + z_dihedral
            
            wing_x_upper[i, j] = x_pos
            wing_y_upper[i, j] = y_twisted
            wing_z_upper[i, j] = z_upper_twisted
            
            wing_x_lower[i, j] = x_pos
            wing_y_lower[i, j] = y_twisted_lower
            wing_z_lower[i, j] = z_lower_twisted
    
    # Professional color scheme based on aircraft type
    if design.aircraft_type == "Fighter":
        upper_colors = [[0, '#4A5568'], [0.5, '#2D3748'], [1, '#1A202C']]  # Dark gray/charcoal
        lower_colors = [[0, '#718096'], [0.5, '#4A5568'], [1, '#2D3748']]  # Lighter gray
    elif design.aircraft_type == "Passenger":
        upper_colors = [[0, '#FFFFFF'], [0.5, '#F7FAFC'], [1, '#EDF2F7']]  # White/silver
        lower_colors = [[0, '#E2E8F0'], [0.5, '#CBD5E0'], [1, '#A0AEC0']]  # Light gray
    else:
        upper_colors = [[0, '#3182CE'], [0.5, '#2B6CB0'], [1, '#2C5282']]  # Blue
        lower_colors = [[0, '#63B3ED'], [0.5, '#4299E1'], [1, '#3182CE']]  # Light blue
    
    # Right wing surfaces with realistic materials
    fig.add_trace(go.Surface(
        x=wing_x_upper, y=wing_y_upper, z=wing_z_upper,
        colorscale=upper_colors,
        showscale=False,
        opacity=0.95,
        lighting=dict(
            ambient=0.3,
            diffuse=0.8,
            specular=0.2,
            roughness=0.1
        ),
        name='Right Wing Upper'
    ))
    
    fig.add_trace(go.Surface(
        x=wing_x_lower, y=wing_y_lower, z=wing_z_lower,
        colorscale=lower_colors,
        showscale=False,
        opacity=0.95,
        lighting=dict(
            ambient=0.3,
            diffuse=0.8,
            specular=0.2,
            roughness=0.1
        ),
        name='Right Wing Lower'
    ))
    
    # Left wing surfaces (mirrored)
    fig.add_trace(go.Surface(
        x=-wing_x_upper, y=wing_y_upper, z=wing_z_upper,
        colorscale=upper_colors,
        showscale=False,
        opacity=0.95,
        lighting=dict(
            ambient=0.3,
            diffuse=0.8,
            specular=0.2,
            roughness=0.1
        ),
        showlegend=False
    ))
    
    fig.add_trace(go.Surface(
        x=-wing_x_lower, y=wing_y_lower, z=wing_z_lower,
        colorscale=lower_colors,
        showscale=False,
        opacity=0.95,
        lighting=dict(
            ambient=0.3,
            diffuse=0.8,
            specular=0.2,
            roughness=0.1
        ),
        showlegend=False
    ))
    

def _add_realistic_fuselage(fig: go.Figure, design: AircraftDesign):
    """Add realistic fuselage with proper lofting and shape"""
    # High-resolution fuselage geometry
    theta = np.linspace(0, 2*np.pi, 36)  # Increased resolution
    n_stations = 80  # More stations for smoother surface
    
    # Fuselage stations from nose to tail
    fuse_y = np.linspace(0.3, -design.fuselage_length * 0.95, n_stations)
    
    # Create realistic fuselage cross-sections
    x_fuse = []
    y_fuse = []
    z_fuse = []
    
    for i, y in enumerate(fuse_y):
        progress = i / (n_stations - 1)
        
        # Realistic fuselage shape profile (Haack series approximation)
        if progress < 0.1:  # Nose section - pointed
            radius_factor = (progress / 0.1) ** 1.5
        elif progress < 0.15:  # Nose to cockpit transition
            radius_factor = 0.5 + 0.5 * (progress - 0.1) / 0.05
        elif progress < 0.75:  # Main fuselage - constant or slight taper
            if design.aircraft_type == "Passenger":
                radius_factor = 1.0  # Constant radius for passenger
            else:
                radius_factor = 1.0 - 0.1 * (progress - 0.15) / 0.6  # Slight taper
        elif progress < 0.9:  # Tail transition
            radius_factor = (1.0 - 0.1) * (0.9 - progress) / 0.15
        else:  # Tail cone - tapered to point
            radius_factor = 0.8 * (1.0 - progress) / 0.1
        
        # Apply realistic scaling
        radius_factor = max(radius_factor, 0.02)  # Minimum radius for pointed ends
        base_radius = design.fuselage_diameter / 2
        
        # Different cross-sectional shapes based on aircraft type
        if design.aircraft_type == "Fighter":
            # More angular, less circular for fighter
            x_circle = []
            z_circle = []
            for angle in theta:
                if abs(np.cos(angle)) > 0.7:  # Top/bottom flattening
                    radius = base_radius * radius_factor * 0.8
                else:
                    radius = base_radius * radius_factor
                x_circle.append(radius * np.cos(angle))
                z_circle.append(radius * np.sin(angle) * 0.9)  # Slightly flattened
            x_circle = np.array(x_circle)
            z_circle = np.array(z_circle)
        else:
            # Circular cross-section for passenger/general aviation
            radius = base_radius * radius_factor
            x_circle = radius * np.cos(theta)
            z_circle = radius * np.sin(theta)
        
        y_circle = np.full_like(x_circle, y)
        
        x_fuse.append(x_circle)
        y_fuse.append(y_circle)
        z_fuse.append(z_circle)
    
    # Convert to numpy arrays
    x_fuse = np.array(x_fuse)
    y_fuse = np.array(y_fuse)
    z_fuse = np.array(z_fuse)
    
    # Professional fuselage colors based on aircraft type
    if design.aircraft_type == "Fighter":
        fuse_colors = [[0, '#2D3748'], [0.5, '#4A5568'], [1, '#718096']]  # Military gray
    elif design.aircraft_type == "Passenger":
        fuse_colors = [[0, '#FFFFFF'], [0.5, '#F7FAFC'], [1, '#EDF2F7']]  # Airline white
    else:
        fuse_colors = [[0, '#4299E1'], [0.5, '#3182CE'], [1, '#2C5282']]  # General aviation blue
    
    # Create fuselage surface with professional materials
    fig.add_trace(go.Surface(
        x=x_fuse, y=y_fuse, z=z_fuse,
        colorscale=fuse_colors,
        showscale=False,
        opacity=0.95,
        lighting=dict(
            ambient=0.4,
            diffuse=0.7,
            specular=0.3,
            roughness=0.05  # Smooth fuselage finish
        ),
        name='Fuselage'
    ))
    

def _add_realistic_empennage(fig: go.Figure, design: AircraftDesign):
    """Add realistic horizontal and vertical tail surfaces"""
    # Calculate realistic tail dimensions
    h_tail_span = np.sqrt(design.h_tail_area * 3.5)  # Realistic aspect ratio
    h_tail_chord = design.h_tail_area / h_tail_span
    v_tail_height = np.sqrt(design.v_tail_area * 1.8)  # Realistic aspect ratio
    v_tail_chord = design.v_tail_area / v_tail_height
    
    # Horizontal tail sweep (typically less than main wing)
    h_tail_sweep = max(design.sweep_angle * 0.6, 15.0)
    v_tail_sweep = max(design.sweep_angle * 0.8, 20.0)
    
    # Horizontal tail surfaces with realistic airfoil
    n_span_tail = 15
    n_chord_tail = 25
    
    span_stations_h = np.linspace(0, 1, n_span_tail)
    chord_stations_h = np.linspace(0, 1, n_chord_tail)
    
    # Thinner airfoil for tail (8% thickness)
    tail_thickness = 0.08
    y_thickness_tail = 5 * tail_thickness * (0.2969 * np.sqrt(chord_stations_h) - 
                                            0.1260 * chord_stations_h - 
                                            0.3516 * chord_stations_h**2 + 
                                            0.2843 * chord_stations_h**3 - 
                                            0.1015 * chord_stations_h**4)
    
    # Create horizontal tail surfaces
    h_tail_x_upper = np.zeros((n_span_tail, n_chord_tail))
    h_tail_y_upper = np.zeros((n_span_tail, n_chord_tail))
    h_tail_z_upper = np.zeros((n_span_tail, n_chord_tail))
    h_tail_x_lower = np.zeros((n_span_tail, n_chord_tail))
    h_tail_y_lower = np.zeros((n_span_tail, n_chord_tail))
    h_tail_z_lower = np.zeros((n_span_tail, n_chord_tail))
    
    for i, span_frac in enumerate(span_stations_h):
        # Linear taper for tail
        tail_chord_local = h_tail_chord * (1 - span_frac * 0.3)  # 30% taper
        
        # Leading edge sweep
        y_le_tail = -design.tail_arm - span_frac * h_tail_span/2 * np.tan(np.radians(h_tail_sweep))
        
        for j, chord_frac in enumerate(chord_stations_h):
            x_pos = span_frac * h_tail_span/2
            y_pos = y_le_tail - chord_frac * tail_chord_local
            
            h_tail_x_upper[i, j] = x_pos
            h_tail_y_upper[i, j] = y_pos
            h_tail_z_upper[i, j] = y_thickness_tail[j] * tail_chord_local
            
            h_tail_x_lower[i, j] = x_pos
            h_tail_y_lower[i, j] = y_pos
            h_tail_z_lower[i, j] = -y_thickness_tail[j] * tail_chord_local
    
    # Tail colors matching aircraft type
    if design.aircraft_type == "Fighter":
        tail_colors = [[0, '#4A5568'], [1, '#2D3748']]  # Military gray
    elif design.aircraft_type == "Passenger":
        tail_colors = [[0, '#EDF2F7'], [1, '#CBD5E0']]  # Airline colors
    else:
        tail_colors = [[0, '#63B3ED'], [1, '#3182CE']]  # General aviation
    
    # Right horizontal tail
    fig.add_trace(go.Surface(
        x=h_tail_x_upper, y=h_tail_y_upper, z=h_tail_z_upper,
        colorscale=tail_colors,
        showscale=False,
        opacity=0.9,
        lighting=dict(ambient=0.3, diffuse=0.7, specular=0.2),
        name='H-Tail Upper'
    ))
    
    fig.add_trace(go.Surface(
        x=h_tail_x_lower, y=h_tail_y_lower, z=h_tail_z_lower,
        colorscale=tail_colors,
        showscale=False,
        opacity=0.9,
        lighting=dict(ambient=0.3, diffuse=0.7, specular=0.2),
        showlegend=False
    ))
    
    # Left horizontal tail (mirrored)
    fig.add_trace(go.Surface(
        x=-h_tail_x_upper, y=h_tail_y_upper, z=h_tail_z_upper,
        colorscale=tail_colors,
        showscale=False,
        opacity=0.9,
        lighting=dict(ambient=0.3, diffuse=0.7, specular=0.2),
        showlegend=False
    ))
    
    fig.add_trace(go.Surface(
        x=-h_tail_x_lower, y=h_tail_y_lower, z=h_tail_z_lower,
        colorscale=tail_colors,
        showscale=False,
        opacity=0.9,
        lighting=dict(ambient=0.3, diffuse=0.7, specular=0.2),
        showlegend=False
    ))
    
    # Vertical tail with realistic shape
    v_tail_x = np.zeros((n_chord_tail, n_span_tail))
    v_tail_y = np.zeros((n_chord_tail, n_span_tail))
    v_tail_z = np.zeros((n_chord_tail, n_span_tail))
    
    height_stations = np.linspace(0, 1, n_span_tail)
    
    for i, chord_frac in enumerate(chord_stations_h):
        for j, height_frac in enumerate(height_stations):
            # Taper in chord and sweep
            v_chord_local = v_tail_chord * (1 - height_frac * 0.4)  # 40% taper
            y_le_v = -design.tail_arm - height_frac * v_tail_height * np.tan(np.radians(v_tail_sweep))
            
            v_tail_x[i, j] = y_thickness_tail[i] * v_chord_local * 0.5  # Half thickness for vertical
            v_tail_y[i, j] = y_le_v - chord_frac * v_chord_local
            v_tail_z[i, j] = height_frac * v_tail_height
    
    fig.add_trace(go.Surface(
        x=v_tail_x, y=v_tail_y, z=v_tail_z,
        colorscale=tail_colors,
        showscale=False,
        opacity=0.9,
        lighting=dict(ambient=0.3, diffuse=0.7, specular=0.2),
        name='V-Tail'
    ))


def _add_realistic_engines(fig: go.Figure, design: AircraftDesign):
    """Add realistic engine nacelles for multi-engine aircraft"""
    # Realistic engine sizing based on aircraft type
    if design.aircraft_type == "Passenger":
        nacelle_diameter = design.fuselage_diameter * 0.4
        nacelle_length = design.fuselage_length * 0.25
        engine_positions = [-design.wingspan * 0.35, design.wingspan * 0.35]
        engine_z_offset = -nacelle_diameter * 0.6  # Below wing
    else:
        nacelle_diameter = design.fuselage_diameter * 0.25
        nacelle_length = design.fuselage_length * 0.2
        engine_positions = [-design.wingspan * 0.25, design.wingspan * 0.25]
        engine_z_offset = -nacelle_diameter * 0.3
    
    # Engine mounting position on wing
    engine_y = -design.wingspan * 0.2
    
    for eng_x in engine_positions:
        # High-resolution nacelle geometry
        theta_eng = np.linspace(0, 2*np.pi, 24)
        n_eng_stations = 30
        y_eng = np.linspace(engine_y + nacelle_length * 0.1, engine_y - nacelle_length * 0.9, n_eng_stations)
        
        x_nacelle = []
        y_nacelle = []
        z_nacelle = []
        
        for i, y in enumerate(y_eng):
            progress = i / (n_eng_stations - 1)
            
            # Realistic nacelle shape (tapered ends)
            if progress < 0.15:  # Nose
                radius_factor = progress / 0.15 * 0.7
            elif progress < 0.85:  # Main body
                radius_factor = 0.7 + 0.3 * np.sin(np.pi * (progress - 0.15) / 0.7)
            else:  # Tail
                radius_factor = 1.0 * (1 - progress) / 0.15
            
            radius = nacelle_diameter/2 * max(radius_factor, 0.1)
            
            x_circle = eng_x + radius * np.cos(theta_eng)
            z_circle = engine_z_offset + radius * np.sin(theta_eng)
            y_circle = np.full_like(x_circle, y)
            
            x_nacelle.append(x_circle)
            y_nacelle.append(y_circle)
            z_nacelle.append(z_circle)
        
        x_nacelle = np.array(x_nacelle)
        y_nacelle = np.array(y_nacelle)
        z_nacelle = np.array(z_nacelle)
        
        # Engine colors
        engine_colors = [[0, '#A0AEC0'], [0.5, '#718096'], [1, '#4A5568']]  # Metallic silver
        
        fig.add_trace(go.Surface(
            x=x_nacelle, y=y_nacelle, z=z_nacelle,
            colorscale=engine_colors,
            showscale=False,
            opacity=0.95,
            lighting=dict(
                ambient=0.2,
                diffuse=0.8,
                specular=0.4,
                roughness=0.1
            ),
            name='Engine' if eng_x == engine_positions[0] else None,
            showlegend=eng_x == engine_positions[0]
        ))


def _configure_professional_layout(fig: go.Figure, design: AircraftDesign):
    """Configure professional layout, lighting and camera"""
    fig.update_layout(
        title=dict(
            text=f"{design.aircraft_type} Aircraft - Professional 3D Model",
            x=0.5,
            font=dict(size=20, color='#2D3748', family="Arial Black")
        ),
        scene=dict(
            xaxis_title="Span (m)",
            yaxis_title="Length (m)", 
            zaxis_title="Height (m)",
            aspectmode='data',
            bgcolor='#F7FAFC',  # Professional light background
            camera=dict(
                eye=dict(x=1.5, y=1.8, z=1.0),
                center=dict(x=0, y=-design.fuselage_length/3, z=0),
                up=dict(x=0, y=0, z=1)
            ),
            xaxis=dict(
                backgroundcolor="#FFFFFF",
                gridcolor="#E2E8F0",
                showbackground=True,
                zeroline=False,
                showticklabels=True,
                title_font=dict(size=14)
            ),
            yaxis=dict(
                backgroundcolor="#FFFFFF", 
                gridcolor="#E2E8F0",
                showbackground=True,
                zeroline=False,
                showticklabels=True,
                title_font=dict(size=14)
            ),
            zaxis=dict(
                backgroundcolor="#FFFFFF",
                gridcolor="#E2E8F0", 
                showbackground=True,
                zeroline=False,
                showticklabels=True,
                title_font=dict(size=14)
            )
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#E2E8F0',
            borderwidth=1
        )
    )
    
    # Enhanced layout with better camera and lighting
    fig.update_layout(
        title=dict(
            text=f"{design.aircraft_type} Aircraft - Enhanced 3D Model",
            x=0.5,
            font=dict(size=18, color='darkblue')
        ),
        scene=dict(
            xaxis_title="Span (m)",
            yaxis_title="Length (m)", 
            zaxis_title="Height (m)",
            aspectmode='data',
            bgcolor='lightblue',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.2),
                center=dict(x=0, y=-design.fuselage_length/3, z=0)
            ),
            xaxis=dict(
                backgroundcolor="white",
                gridcolor="lightgray",
                showbackground=True,
                zeroline=False
            ),
            yaxis=dict(
                backgroundcolor="white", 
                gridcolor="lightgray",
                showbackground=True,
                zeroline=False
            ),
            zaxis=dict(
                backgroundcolor="white",
                gridcolor="lightgray", 
                showbackground=True,
                zeroline=False
            )
        ),
        showlegend=True,
        height=800,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig


def export_to_stl(design: AircraftDesign) -> bytes:
    """
    Export aircraft geometry to STL format
    """
    stl_content = "solid aircraft\n"
    
    # Simplified STL generation - create basic triangulated surfaces
    # Wing triangles
    wing_vertices = [
        [0, 0, 0],
        [design.wingspan/2, -design.wingspan/2 * np.tan(np.radians(design.sweep_angle)), 0],
        [design.wingspan/2, -design.wingspan/2 * np.tan(np.radians(design.sweep_angle)) - design.tip_chord, 0],
        [0, -design.root_chord, 0]
    ]
    
    # Add triangles for wing
    stl_content += f"  facet normal 0 0 1\n"
    stl_content += f"    outer loop\n"
    for v in wing_vertices[:3]:
        stl_content += f"      vertex {v[0]} {v[1]} {v[2]}\n"
    stl_content += f"    endloop\n"
    stl_content += f"  endfacet\n"
    
    stl_content += "endsolid aircraft\n"
    
    return stl_content.encode()


def generate_design_report(design: AircraftDesign) -> str:
    """
    Generate comprehensive design report
    """
    report = f"""
# Aircraft Design Report
## {design.aircraft_type} Aircraft Design

### Executive Summary
- **MTOW**: {design.mtow:.1f} kg
- **Range**: {design.range_km:.1f} km
- **L/D Cruise**: {design.ld_cruise:.1f}
- **Wing Area**: {design.wing_area:.1f} m²

### Weight Breakdown
- **MTOW**: {design.mtow:.1f} kg
- **OEW**: {design.oew:.1f} kg
- **Fuel Weight**: {design.fuel_weight:.1f} kg
- **Payload**: {design.payload_weight:.1f} kg

### Wing Geometry
- **Wing Area**: {design.wing_area:.2f} m²
- **Wingspan**: {design.wingspan:.2f} m
- **Aspect Ratio**: {design.aspect_ratio:.2f}
- **Sweep Angle**: {design.sweep_angle:.1f}°
- **Taper Ratio**: {design.taper_ratio:.2f}
- **Root Chord**: {design.root_chord:.2f} m
- **Tip Chord**: {design.tip_chord:.2f} m
- **MAC**: {design.mac:.2f} m

### Aerodynamics
- **CL Cruise**: {design.cl_cruise:.3f}
- **CL Max**: {design.cl_max:.2f}
- **CD0**: {design.cd0:.4f}
- **L/D Max**: {design.ld_max:.1f}
- **L/D Cruise**: {design.ld_cruise:.1f}

### Propulsion
- **Thrust Required**: {design.thrust_required:.0f} N
- **Engine Count**: {design.engine_count}
- **SFC**: {design.sfc*3600:.3f} kg/N/hr

### Performance
- **Range**: {design.range_km:.1f} km
- **Endurance**: {design.endurance_hrs:.1f} hours
- **Cruise Speed**: {design.cruise_speed:.1f} m/s
- **Stall Speed**: {design.stall_speed:.1f} m/s
- **Climb Rate**: {design.climb_rate:.1f} m/s
- **Service Ceiling**: {design.ceiling:.0f} m
- **Takeoff Distance**: {design.takeoff_distance:.0f} m

### Stability
- **Static Margin**: {design.static_margin:.3f}
- **CG Range**: {design.cg_range[0]:.2f} to {design.cg_range[1]:.2f} MAC

### Design Convergence
- **Iterations**: {len(design.iterations)}
- **Converged**: {'Yes' if design.converged else 'No'}
"""
    
    return report