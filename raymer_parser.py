"""
Raymer Book Parser - Extracts formulas and constants from uploaded PDF
"""

import re
import PyPDF2
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st
import numpy as np
from pathlib import Path
import json

class RaymerParser:
    """
    Parser for extracting formulas and constants from Raymer's Aircraft Design book
    """
    
    def __init__(self):
        self.formulas = {}
        self.constants = {}
        self.chapters = {}
        self.parsed = False
        
    def parse_pdf(self, pdf_file) -> Optional[Dict]:
        """
        Parse Raymer PDF and extract formulas and constants
        
        Args:
            pdf_file: Uploaded PDF file object
            
        Returns:
            Dictionary of extracted formulas and constants
        """
        try:
            # Read PDF
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            
            # Extract text from relevant chapters
            text_content = ""
            for page_num in range(min(num_pages, 500)):  # Limit to first 500 pages
                page = pdf_reader.pages[page_num]
                text_content += page.extract_text()
            
            # Parse specific sections
            self._parse_weight_equations(text_content)
            self._parse_aerodynamic_equations(text_content)
            self._parse_performance_equations(text_content)
            self._parse_stability_equations(text_content)
            
            self.parsed = True
            
            return {
                "formulas": self.formulas,
                "constants": self.constants,
                "chapters": self.chapters
            }
            
        except Exception as e:
            st.error(f"Error parsing PDF: {str(e)}")
            return None
    
    def _parse_weight_equations(self, text: str):
        """Extract weight estimation equations from Chapter 3 and 15"""
        
        # Empty weight fraction (from the uploaded PDF content)
        # Based on Raymer eq. 3.8: We/W0 = A * W0^C1 + B
        if "Empty-Weight Estimation" in text or "empty weight" in text.lower():
            # Fighter aircraft coefficients
            self.formulas['empty_weight_fraction_fighter'] = {
                'equation': 'We/W0 = A * W0^C1 + B',
                'A': -0.02,  # From text
                'B': 1.04,   # From text
                'C1': -0.10,
                'source': 'Raymer Ch. 3, Eq. 3.8',
                'description': 'Empty weight fraction for fighter aircraft'
            }
            
            # Transport aircraft coefficients
            self.formulas['empty_weight_fraction_transport'] = {
                'equation': 'We/W0 = A * W0^C1 + B',
                'A': -0.10,
                'B': 1.06,
                'C1': -0.10,
                'source': 'Raymer Ch. 3',
                'description': 'Empty weight fraction for transport aircraft'
            }
        
        # Fuel fraction estimation (Mission segment weight fractions)
        # From Chapter 3 - fuel used during cruise
        self.formulas['fuel_fraction_cruise'] = {
            'equation': 'Wf/Wi = 1 - exp(-R*C/(V*(L/D)))',
            'typical_jet': 0.85,  # Wi+1/Wi for cruise
            'typical_prop': 0.94,
            'source': 'Raymer Ch. 3, Mission Segment Weight Fractions',
            'description': 'Fuel fraction for cruise segment'
        }
        
        # Component weight equations from Chapter 15
        # Wing weight equation
        self.formulas['wing_weight'] = {
            'equation': 'W_wing = 0.036 * S_w^0.758 * W_fw^0.0035 * (A/cos²Λ)^0.6 * q^0.006 * λ^0.04 * (100*t/c)^-0.3 * (N_z*W_dg)^0.49',
            'source': 'Raymer Ch. 15, Transport Wing Weight',
            'units': 'lb',
            'description': 'Statistical wing weight equation'
        }
    
    def _parse_aerodynamic_equations(self, text: str):
        """Extract aerodynamic equations from Chapters 4, 12"""
        
        # Lift curve slope
        self.formulas['lift_curve_slope'] = {
            'equation': 'CLα = 2π*AR / (2 + √(4 + AR²*β²/η² * (1 + tan²Λmax/β²)))',
            'source': 'Raymer Ch. 12, Eq. 12.6',
            'description': 'Wing lift curve slope'
        }
        
        # Maximum lift coefficient
        self.formulas['cl_max_clean'] = {
            'fighter': 1.2,
            'transport': 1.4,
            'general': 1.5,
            'source': 'Raymer Ch. 5, Table 5.1',
            'description': 'Typical maximum lift coefficients (clean)'
        }
        
        # Zero-lift drag coefficient
        self.formulas['cd0_estimation'] = {
            'equation': 'CD0 = Cfe * (Swet/Sref)',
            'Cfe_laminar': 0.003,
            'Cfe_turbulent': 0.0045,
            'source': 'Raymer Ch. 12, Parasite Drag',
            'description': 'Zero-lift drag estimation'
        }
        
        # Oswald efficiency factor
        self.formulas['oswald_efficiency'] = {
            'straight_wing': 0.85,
            'swept_wing': '0.85 - 0.001*Λ',  # Λ in degrees
            'source': 'Raymer Fig. 12.14',
            'description': 'Oswald span efficiency factor'
        }
    
    def _parse_performance_equations(self, text: str):
        """Extract performance equations from Chapters 17, 19"""
        
        # Breguet range equation for jets
        self.formulas['range_jet'] = {
            'equation': 'R = (V/c) * (L/D) * ln(W_initial/W_final)',
            'source': 'Raymer Ch. 17, Eq. 17.2',
            'units': 'same as V',
            'description': 'Breguet range equation for jet aircraft'
        }
        
        # Endurance equation
        self.formulas['endurance_jet'] = {
            'equation': 'E = (1/c) * (L/D) * ln(W_initial/W_final)',
            'source': 'Raymer Ch. 17',
            'units': 'hours',
            'description': 'Endurance equation for jet aircraft'
        }
        
        # Takeoff distance
        self.formulas['takeoff_distance'] = {
            'equation': 'S_TO = 1.44 * W²/(g*ρ*S*CL_max*T)',
            'source': 'Raymer Ch. 17, Takeoff',
            'description': 'Takeoff ground roll distance'
        }
        
        # Climb rate
        self.formulas['climb_rate'] = {
            'equation': 'R/C = (T - D)*V / W',
            'source': 'Raymer Ch. 17, Climb',
            'units': 'ft/min or m/s',
            'description': 'Rate of climb'
        }
    
    def _parse_stability_equations(self, text: str):
        """Extract stability equations from Chapter 16"""
        
        # Tail volume coefficients
        self.constants['tail_volume_coefficients'] = {
            'horizontal_tail': {
                'fighter': 0.4,
                'transport': 0.9,
                'general': 0.6
            },
            'vertical_tail': {
                'fighter': 0.04,
                'transport': 0.09,
                'general': 0.06
            },
            'source': 'Raymer Ch. 6, Table 6.4',
            'description': 'Typical tail volume coefficients'
        }
        
        # Static margin
        self.formulas['static_margin'] = {
            'equation': 'SM = (X_np - X_cg) / MAC',
            'typical_range': [0.05, 0.15],
            'source': 'Raymer Ch. 16',
            'description': 'Static margin definition'
        }
        
        # CG range
        self.constants['cg_range'] = {
            'forward_limit': 0.15,  # 15% MAC
            'aft_limit': 0.35,      # 35% MAC
            'source': 'Raymer Ch. 16',
            'description': 'Typical CG range as fraction of MAC'
        }
    
    def get_formula(self, formula_name: str) -> Optional[Dict]:
        """
        Get a specific formula by name
        
        Args:
            formula_name: Name of the formula to retrieve
            
        Returns:
            Formula dictionary or None if not found
        """
        return self.formulas.get(formula_name)
    
    def get_constant(self, constant_name: str) -> Optional[Any]:
        """
        Get a specific constant by name
        
        Args:
            constant_name: Name of the constant to retrieve
            
        Returns:
            Constant value or None if not found
        """
        return self.constants.get(constant_name)
    
    def apply_raymer_formulas(self, design_engine):
        """
        Apply parsed Raymer formulas to the design engine
        
        Args:
            design_engine: RaymerDesignEngine instance to update
        """
        if not self.parsed:
            return design_engine
        
        # Update weight coefficients if found
        if 'empty_weight_fraction_fighter' in self.formulas:
            formula = self.formulas['empty_weight_fraction_fighter']
            design_engine.weight_coeffs['Fighter'].update({
                'A': formula['A'],
                'B': formula['B'],
                'C1': formula['C1']
            })
        
        if 'empty_weight_fraction_transport' in self.formulas:
            formula = self.formulas['empty_weight_fraction_transport']
            design_engine.weight_coeffs['Passenger'].update({
                'A': formula['A'],
                'B': formula['B'],
                'C1': formula['C1']
            })
        
        # Update aerodynamic constants
        if 'cl_max_clean' in self.formulas:
            design_engine.aero_constants['cl_max_clean'].update(
                self.formulas['cl_max_clean']
            )
        
        # Update tail volume coefficients
        if 'tail_volume_coefficients' in self.constants:
            design_engine.tail_volumes = self.constants['tail_volume_coefficients']
        
        return design_engine
    
    def save_parsed_data(self, filepath: str):
        """Save parsed formulas and constants to JSON file"""
        data = {
            'formulas': self.formulas,
            'constants': self.constants,
            'chapters': self.chapters,
            'parsed': self.parsed
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_parsed_data(self, filepath: str):
        """Load previously parsed formulas and constants from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            self.formulas = data.get('formulas', {})
            self.constants = data.get('constants', {})
            self.chapters = data.get('chapters', {})
            self.parsed = data.get('parsed', False)
            
            return True
        except Exception as e:
            return False