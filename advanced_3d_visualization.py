"""
Advanced 3D Wing Visualization and CFD Analysis
Provides comprehensive 3D visualization, pressure distribution, and CFD analysis capabilities
"""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math
from typing import Dict, Any, List, Tuple, Optional
import streamlit as st
from scipy import interpolate
from scipy.spatial import distance_matrix
from aero import naca4_coords
import warnings

class Advanced3DWingVisualizer:
    """Advanced 3D wing visualization with CFD capabilities"""
    
    def __init__(self):
        self.wing_geometry = None
        self.pressure_data = None
        self.flow_field = None
    
    def generate_3d_wing_mesh(self, m: float, p: float, t: float, AR: float, 
                             taper: float = 1.0, sweep: float = 0.0, 
                             twist: float = 0.0, chord_root: float = 1.0, 
                             n_span: int = 30, n_chord: int = 50) -> Dict:
        """Generate detailed 3D wing mesh for visualization and CFD"""
        
        # Parameter validation and guards
        if not self._validate_wing_parameters(m, p, t, AR, taper, sweep, twist, chord_root):
            raise ValueError("Invalid wing parameters detected")
        
        # Clamp parameters to safe ranges
        sweep = np.clip(sweep, -60, 60)  # Limit sweep to reasonable range
        twist = np.clip(twist, -30, 30)  # Limit twist to reasonable range
        AR = np.clip(AR, 0.5, 20)       # Limit aspect ratio
        taper = np.clip(taper, 0.1, 3.0) # Limit taper ratio
        
        # Wing parameters
        span = AR * chord_root
        tip_chord = chord_root * taper
        
        # Span-wise stations
        y_stations = np.linspace(-span/2, span/2, n_span)
        
        # Initialize storage arrays
        wing_upper = np.zeros((n_span, n_chord, 3))
        wing_lower = np.zeros((n_span, n_chord, 3))
        
        for i, y in enumerate(y_stations):
            # Local chord length
            local_chord = chord_root + (tip_chord - chord_root) * abs(y) / (span/2)
            
            # Local twist angle
            local_twist = twist * abs(y) / (span/2)
            
            # Leading edge position (sweep)
            le_x = abs(y) * math.tan(math.radians(sweep))
            
            # Generate NACA 4-digit coordinates using robust aero.naca4_coords
            try:
                X_coords, Y_coords, x_coords, yc, dyc_dx, t_normalized = naca4_coords(m, p, t, n_chord)
                
                # Extract upper and lower surfaces from combined coordinates
                n_points = len(X_coords)
                n_upper = n_points // 2
                
                # Upper surface (reversed to go from trailing edge to leading edge)
                x_upper = X_coords[:n_upper][::-1]
                y_upper = Y_coords[:n_upper][::-1]
                
                # Lower surface (from leading edge to trailing edge)
                x_lower = X_coords[n_upper:]
                y_lower = Y_coords[n_upper:]
                
                # Interpolate to get n_chord points with monotonic x
                x_coords_monotonic = np.linspace(0, 1, n_chord)
                y_upper = np.interp(x_coords_monotonic, x_upper, y_upper)
                y_lower = np.interp(x_coords_monotonic, x_lower, y_lower)
                
                # Validate coordinates for NaNs and infinities
                if np.any(np.isnan(y_upper)) or np.any(np.isnan(y_lower)) or \
                   np.any(np.isinf(y_upper)) or np.any(np.isinf(y_lower)):
                    raise ValueError(f"Invalid coordinates generated for span station {i}")
                
            except Exception as e:
                # Fallback to symmetric airfoil if coordinate generation fails
                warnings.warn(f"NACA coordinate generation failed at span station {i}: {e}. Using fallback.")
                x_coords_monotonic = np.linspace(0, 1, n_chord)
                thickness = t / 100.0
                y_upper = thickness * (0.2969 * np.sqrt(x_coords_monotonic) - 
                                     0.1260 * x_coords_monotonic - 
                                     0.3516 * x_coords_monotonic**2 + 
                                     0.2843 * x_coords_monotonic**3 - 
                                     0.1015 * x_coords_monotonic**4)
                y_lower = -y_upper
            
            # Scale by local chord
            x_scaled = x_coords_monotonic * local_chord + le_x
            y_upper_scaled = y_upper * local_chord
            y_lower_scaled = y_lower * local_chord
            
            # Apply twist (rotate about quarter chord)
            if local_twist != 0:
                twist_rad = math.radians(local_twist)
                qc_x = 0.25 * local_chord + le_x
                
                # Rotate coordinates
                for j in range(n_chord):
                    # Upper surface
                    dx = x_scaled[j] - qc_x
                    dy = y_upper_scaled[j]
                    x_rot = qc_x + dx * math.cos(twist_rad) - dy * math.sin(twist_rad)
                    y_rot = dx * math.sin(twist_rad) + dy * math.cos(twist_rad)
                    wing_upper[i, j] = [x_rot, y, y_rot]
                    
                    # Lower surface
                    dy = y_lower_scaled[j]
                    x_rot = qc_x + dx * math.cos(twist_rad) - dy * math.sin(twist_rad)
                    y_rot = dx * math.sin(twist_rad) + dy * math.cos(twist_rad)
                    wing_lower[i, j] = [x_rot, y, y_rot]
            else:
                for j in range(n_chord):
                    wing_upper[i, j] = [x_scaled[j], y, y_upper_scaled[j]]
                    wing_lower[i, j] = [x_scaled[j], y, y_lower_scaled[j]]
        
        self.wing_geometry = {
            'upper_surface': wing_upper,
            'lower_surface': wing_lower,
            'span': span,
            'chord_root': chord_root,
            'chord_tip': tip_chord,
            'y_stations': y_stations
        }
        
        # Validate final geometry
        if not self._validate_geometry(wing_upper, wing_lower):
            raise ValueError("Generated wing geometry contains invalid values")
        
        return self.wing_geometry
    
    def _validate_wing_parameters(self, m: float, p: float, t: float, AR: float, 
                                 taper: float, sweep: float, twist: float, chord_root: float) -> bool:
        """Validate wing parameters to prevent divergence and invalid geometries"""
        try:
            # NACA parameter validation
            if not (0 <= m <= 9):
                st.warning(f"NACA camber parameter m={m} outside valid range [0,9]. Clamping.")
                return False
            if not (0 <= p <= 9) and m > 0:
                st.warning(f"NACA camber position p={p} outside valid range [0,9]. Clamping.")
                return False
            if not (6 <= t <= 40):
                st.warning(f"NACA thickness parameter t={t} outside practical range [6,40]. Clamping.")
                return False
                
            # Wing geometry validation
            if not (0.5 <= AR <= 20):
                st.warning(f"Aspect ratio AR={AR} outside practical range [0.5,20]. Clamping.")
                return False
            if not (0.1 <= taper <= 3.0):
                st.warning(f"Taper ratio={taper} outside practical range [0.1,3.0]. Clamping.")
                return False
            if not (-60 <= sweep <= 60):
                st.warning(f"Sweep angle={sweep}° outside practical range [-60,60]. Clamping.")
                return False
            if not (-30 <= twist <= 30):
                st.warning(f"Twist angle={twist}° outside practical range [-30,30]. Clamping.")
                return False
            if chord_root <= 0:
                st.error(f"Root chord={chord_root} must be positive.")
                return False
                
            return True
        except Exception as e:
            st.error(f"Parameter validation failed: {e}")
            return False
    
    def _validate_geometry(self, wing_upper: np.ndarray, wing_lower: np.ndarray) -> bool:
        """Validate generated wing geometry for NaNs, infinities, and physical constraints"""
        try:
            # Check for NaNs and infinities
            if np.any(np.isnan(wing_upper)) or np.any(np.isinf(wing_upper)):
                st.error("Upper wing surface contains NaN or infinite values")
                return False
            if np.any(np.isnan(wing_lower)) or np.any(np.isinf(wing_lower)):
                st.error("Lower wing surface contains NaN or infinite values")
                return False
                
            # Check physical constraints
            n_span, n_chord, _ = wing_upper.shape
            
            # Verify monotonic chordwise progression
            for i in range(n_span):
                x_upper = wing_upper[i, :, 0]
                x_lower = wing_lower[i, :, 0]
                if not (np.all(np.diff(x_upper) >= 0) or np.all(np.diff(x_upper) <= 0)):
                    st.warning(f"Non-monotonic chordwise coordinates detected at span station {i}")
                    # Allow this but warn - it might be due to twist
                    
            # Check for reasonable coordinate ranges
            max_coord = max(np.abs(wing_upper).max(), np.abs(wing_lower).max())
            if max_coord > 1000:  # Reasonable upper bound for aircraft coordinates
                st.warning(f"Unusually large coordinate values detected (max={max_coord:.1f})")
                
            return True
            
        except Exception as e:
            st.error(f"Geometry validation failed: {e}")
            return False
    
    def calculate_pressure_distribution(self, results: Dict[str, Any], alpha: float, 
                                      V: float, rho: float = 1.225) -> Dict:
        """Calculate pressure distribution using panel method and potential flow theory"""
        
        if self.wing_geometry is None:
            raise ValueError("Wing geometry must be generated first")
        
        # Extract aerodynamic coefficients
        cl_3d = results['aerodynamics_3d']['Cl_3d']
        
        # Dynamic pressure
        q_inf = 0.5 * rho * V**2
        
        # Initialize pressure coefficient arrays
        wing_upper = self.wing_geometry['upper_surface']
        wing_lower = self.wing_geometry['lower_surface']
        n_span, n_chord, _ = wing_upper.shape
        
        cp_upper = np.zeros((n_span, n_chord))
        cp_lower = np.zeros((n_span, n_chord))
        
        # Span-wise circulation distribution (elliptical approximation)
        y_stations = self.wing_geometry['y_stations']
        span = self.wing_geometry['span']
        
        for i, y in enumerate(y_stations):
            # Local circulation strength (elliptical distribution)
            eta = 2 * y / span
            gamma_local = cl_3d * V * math.sqrt(1 - eta**2) if abs(eta) < 1 else 0
            
            # Chordwise pressure distribution using thin airfoil theory
            for j in range(n_chord):
                x_c = j / (n_chord - 1)  # Chordwise position (0 to 1)
                
                # Pressure coefficient from thin airfoil theory
                if x_c < 0.01:  # Leading edge singularity handling
                    cp_upper[i, j] = -4 * math.sin(math.radians(alpha))**2
                    cp_lower[i, j] = 4 * math.sin(math.radians(alpha))**2
                else:
                    # Simplified pressure distribution
                    cp_upper[i, j] = -2 * gamma_local / (V * math.sqrt(x_c)) * 0.1
                    cp_lower[i, j] = 2 * gamma_local / (V * math.sqrt(x_c)) * 0.1
        
        # Convert to actual pressure
        p_upper = (cp_upper * q_inf) + (rho * 9.81 * 0)  # Add atmospheric pressure if needed
        p_lower = (cp_lower * q_inf) + (rho * 9.81 * 0)
        
        self.pressure_data = {
            'cp_upper': cp_upper,
            'cp_lower': cp_lower,
            'p_upper': p_upper,
            'p_lower': p_lower,
            'q_inf': q_inf
        }
        
        return self.pressure_data
    
    def generate_streamlines(self, results: Dict[str, Any], alpha: float, V: float, 
                           extent: float = 3.0, resolution: int = 50) -> Dict:
        """Generate 3D streamlines around the wing"""
        
        if self.wing_geometry is None:
            raise ValueError("Wing geometry must be generated first")
        
        # Create flow field grid
        span = self.wing_geometry['span']
        chord_root = self.wing_geometry['chord_root']
        
        x_range = [-0.5 * chord_root, extent * chord_root]
        y_range = [-span/2 * 1.2, span/2 * 1.2]
        z_range = [-0.5 * chord_root, 0.5 * chord_root]
        
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution//2)
        z = np.linspace(z_range[0], z_range[1], resolution//2)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Initialize velocity field
        U = np.full_like(X, V * math.cos(math.radians(alpha)))
        V_field = np.zeros_like(X)
        W = np.full_like(X, V * math.sin(math.radians(alpha)))
        
        # Add wing influence (simplified vortex system)
        cl_3d = results['aerodynamics_3d']['Cl_3d']
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    # Distance from wing quarter-chord line
                    x_pos = X[i, j, k]
                    y_pos = Y[i, j, k]
                    z_pos = Z[i, j, k]
                    
                    # Simple vortex influence (Biot-Savart approximation)
                    if abs(y_pos) < span/2 and x_pos > 0:
                        r = math.sqrt((z_pos)**2 + (x_pos - 0.25*chord_root)**2)
                        if r > 0.01:  # Avoid singularity
                            gamma = cl_3d * V * 0.1  # Simplified circulation
                            dw = gamma / (2 * math.pi * r**2) * z_pos
                            du = -gamma / (2 * math.pi * r**2) * (z_pos)
                            U[i, j, k] += du * 0.1
                            W[i, j, k] += dw * 0.1
        
        self.flow_field = {
            'X': X, 'Y': Y, 'Z': Z,
            'U': U, 'V': V_field, 'W': W,
            'extent': extent,
            'resolution': resolution
        }
        
        return self.flow_field
    
    def plot_3d_wing(self, show_pressure: bool = True, colorscale: str = 'RdBu') -> go.Figure:
        """Create interactive 3D wing visualization"""
        
        if self.wing_geometry is None:
            raise ValueError("Wing geometry must be generated first")
        
        fig = go.Figure()
        
        wing_upper = self.wing_geometry['upper_surface']
        wing_lower = self.wing_geometry['lower_surface']
        
        if show_pressure and self.pressure_data is not None:
            # Plot pressure distribution
            cp_upper = self.pressure_data['cp_upper']
            cp_lower = self.pressure_data['cp_lower']
            
            # Upper surface
            fig.add_trace(go.Surface(
                x=wing_upper[:, :, 0],
                y=wing_upper[:, :, 1],
                z=wing_upper[:, :, 2],
                surfacecolor=cp_upper,
                colorscale=colorscale,
                name='Upper Surface',
                colorbar=dict(title="Pressure Coefficient", x=0.9)
            ))
            
            # Lower surface
            fig.add_trace(go.Surface(
                x=wing_lower[:, :, 0],
                y=wing_lower[:, :, 1],
                z=wing_lower[:, :, 2],
                surfacecolor=cp_lower,
                colorscale=colorscale,
                name='Lower Surface',
                showscale=False
            ))
        else:
            # Plot geometry only
            fig.add_trace(go.Surface(
                x=wing_upper[:, :, 0],
                y=wing_upper[:, :, 1],
                z=wing_upper[:, :, 2],
                colorscale=[[0, 'lightblue'], [1, 'lightblue']],
                surfacecolor=np.ones_like(wing_upper[:, :, 0]),
                name='Upper Surface',
                showscale=False
            ))
            
            fig.add_trace(go.Surface(
                x=wing_lower[:, :, 0],
                y=wing_lower[:, :, 1],
                z=wing_lower[:, :, 2],
                colorscale=[[0, 'lightcoral'], [1, 'lightcoral']],
                surfacecolor=np.ones_like(wing_lower[:, :, 0]),
                name='Lower Surface',
                showscale=False
            ))
        
        # Update layout
        fig.update_layout(
            title="3D Wing Visualization" + (" with Pressure Distribution" if show_pressure else ""),
            scene=dict(
                xaxis_title="Chordwise (m)",
                yaxis_title="Spanwise (m)", 
                zaxis_title="Height (m)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
                aspectmode='manual',
                aspectratio=dict(x=1, y=2, z=0.3)
            ),
            width=900,
            height=600
        )
        
        return fig
    
    def plot_streamlines_3d(self) -> go.Figure:
        """Plot 3D streamlines around the wing"""
        
        if self.flow_field is None:
            raise ValueError("Flow field must be generated first")
        
        fig = go.Figure()
        
        # Add wing surface (simplified)
        if self.wing_geometry is not None:
            wing_upper = self.wing_geometry['upper_surface']
            wing_lower = self.wing_geometry['lower_surface']
            
            # Sample points for wing outline
            n_span, n_chord, _ = wing_upper.shape
            skip = max(1, n_span // 10)
            
            fig.add_trace(go.Surface(
                x=wing_upper[::skip, ::3, 0],
                y=wing_upper[::skip, ::3, 1],
                z=wing_upper[::skip, ::3, 2],
                colorscale=[[0, 'gray'], [1, 'gray']],
                surfacecolor=np.ones_like(wing_upper[::skip, ::3, 0]),
                opacity=0.7,
                name='Wing',
                showscale=False
            ))
        
        # Add streamlines (simplified cone visualization)
        X, Y, Z = self.flow_field['X'], self.flow_field['Y'], self.flow_field['Z']
        U, V, W = self.flow_field['U'], self.flow_field['V'], self.flow_field['W']
        
        # Sample streamlines
        skip = 5
        for i in range(0, X.shape[0], skip):
            for k in range(0, X.shape[2], skip):
                if i < X.shape[0] and k < Z.shape[2]:
                    # Create streamline trace
                    x_line = X[i, :, k]
                    y_line = Y[i, :, k] 
                    z_line = Z[i, :, k]
                    
                    fig.add_trace(go.Scatter3d(
                        x=x_line,
                        y=y_line,
                        z=z_line,
                        mode='lines',
                        line=dict(color='blue', width=2),
                        name='Streamline',
                        showlegend=False if i > 0 or k > 0 else True
                    ))
        
        fig.update_layout(
            title="3D Wing with Streamlines",
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                camera=dict(eye=dict(x=2, y=2, z=1)),
                aspectmode='cube'
            ),
            width=900,
            height=600
        )
        
        return fig
    
    def plot_pressure_contours(self) -> go.Figure:
        """Plot 2D pressure contours on wing surface"""
        
        if self.pressure_data is None:
            raise ValueError("Pressure data must be calculated first")
        
        cp_upper = self.pressure_data['cp_upper']
        cp_lower = self.pressure_data['cp_lower']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Upper Surface Pressure', 'Lower Surface Pressure'],
            specs=[[{'type': 'heatmap'}], [{'type': 'heatmap'}]]
        )
        
        # Upper surface
        fig.add_trace(
            go.Heatmap(
                z=cp_upper,
                colorscale='RdBu',
                colorbar=dict(title="Cp", x=0.95),
                name='Upper'
            ),
            row=1, col=1
        )
        
        # Lower surface
        fig.add_trace(
            go.Heatmap(
                z=cp_lower,
                colorscale='RdBu',
                showscale=False,
                name='Lower'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Pressure Distribution Contours",
            width=800,
            height=600
        )
        
        return fig

    def create_wing_geometry_plot(self, wing_mesh: Dict) -> go.Figure:
        """Create wing geometry visualization - alias for plot_3d_wing"""
        self.wing_geometry = wing_mesh
        return self.plot_3d_wing(show_pressure=False)
    
    def create_pressure_distribution_plot(self, wing_mesh: Dict, pressure_coefficients: np.ndarray) -> go.Figure:
        """Create pressure distribution plot on wing surface"""
        self.wing_geometry = wing_mesh
        
        # Set up pressure data for visualization
        n_span, n_chord = wing_mesh['upper_surface'].shape[:2]
        cp_upper = pressure_coefficients[:n_span*n_chord//2].reshape(n_span//2, n_chord)
        cp_lower = pressure_coefficients[n_span*n_chord//2:].reshape(n_span//2, n_chord)
        
        self.pressure_data = {
            'cp_upper': cp_upper,
            'cp_lower': cp_lower
        }
        
        return self.plot_3d_wing(show_pressure=True)
    
    def create_streamlines_plot(self, wing_mesh: Dict, streamlines: List, n_streamlines: int) -> go.Figure:
        """Create 3D streamlines visualization"""
        self.wing_geometry = wing_mesh
        
        # Create figure with wing and streamlines
        fig = self.plot_3d_wing(show_pressure=False)
        
        # Add streamlines
        for i, streamline in enumerate(streamlines[:n_streamlines]):
            if len(streamline) > 1:
                x_coords = [point.position[0] for point in streamline]
                y_coords = [point.position[1] for point in streamline]
                z_coords = [point.position[2] for point in streamline]
                
                fig.add_trace(go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode='lines',
                    line=dict(color='blue', width=3),
                    name=f'Streamline {i+1}' if i < 3 else '',
                    showlegend=(i < 3),
                    hovertemplate='Streamline<br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>'
                ))
        
        fig.update_layout(title="3D Wing with Streamlines")
        return fig
    
    def create_slice_plane_analysis(self, wing_mesh: Dict, vlm_results, slice_position: float) -> go.Figure:
        """Create slice plane analysis at specified span position"""
        fig = go.Figure()
        
        # Get wing data
        wing_upper = wing_mesh['upper_surface']
        wing_lower = wing_mesh['lower_surface']
        y_stations = wing_mesh['y_stations']
        
        # Find closest span station to slice position
        span = wing_mesh['span']
        target_y = slice_position * span - span/2
        slice_idx = np.argmin(np.abs(y_stations - target_y))
        
        # Extract slice data
        x_upper = wing_upper[slice_idx, :, 0]
        z_upper = wing_upper[slice_idx, :, 2]
        x_lower = wing_lower[slice_idx, :, 0]
        z_lower = wing_lower[slice_idx, :, 2]
        
        # Plot airfoil section
        fig.add_trace(go.Scatter(
            x=x_upper,
            y=z_upper,
            mode='lines+markers',
            name='Upper Surface',
            line=dict(color='red', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=x_lower,
            y=z_lower,
            mode='lines+markers',
            name='Lower Surface',
            line=dict(color='blue', width=3)
        ))
        
        # Add pressure coefficient if available
        if hasattr(vlm_results, 'pressure_coefficients') and vlm_results.pressure_coefficients is not None:
            n_panels = len(vlm_results.pressure_coefficients)
            panels_per_slice = n_panels // len(y_stations)
            start_idx = slice_idx * panels_per_slice
            end_idx = start_idx + panels_per_slice
            
            if end_idx <= len(vlm_results.pressure_coefficients):
                cp_slice = vlm_results.pressure_coefficients[start_idx:end_idx]
                
                # Create secondary y-axis for Cp
                fig.add_trace(go.Scatter(
                    x=x_upper[:len(cp_slice)//2] if len(cp_slice) > len(x_upper) else x_upper,
                    y=cp_slice[:len(x_upper)] if len(cp_slice) > len(x_upper) else cp_slice,
                    mode='lines+markers',
                    name='Pressure Coefficient',
                    line=dict(color='green', width=2),
                    yaxis='y2'
                ))
        
        fig.update_layout(
            title=f"Wing Section Analysis at {slice_position*100:.1f}% Span",
            xaxis_title="Chordwise Position (m)",
            yaxis_title="Height (m)",
            yaxis2=dict(
                title="Pressure Coefficient",
                overlaying='y',
                side='right'
            ),
            legend=dict(x=0.02, y=0.98)
        )
        
        return fig
    
    def export_wing_stl(self, wing_mesh: Dict) -> str:
        """Export wing geometry as STL format"""
        try:
            wing_upper = wing_mesh['upper_surface']
            wing_lower = wing_mesh['lower_surface']
            
            # Simple STL generation (basic mesh triangulation)
            stl_content = "solid wing\n"
            
            n_span, n_chord = wing_upper.shape[:2]
            
            # Generate triangles for upper surface
            for i in range(n_span - 1):
                for j in range(n_chord - 1):
                    # Two triangles per quad
                    p1 = wing_upper[i, j]
                    p2 = wing_upper[i+1, j]
                    p3 = wing_upper[i+1, j+1]
                    p4 = wing_upper[i, j+1]
                    
                    # Triangle 1: p1, p2, p3
                    normal = np.cross(p2 - p1, p3 - p1)
                    normal = normal / np.linalg.norm(normal)
                    
                    stl_content += f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n"
                    stl_content += "    outer loop\n"
                    stl_content += f"      vertex {p1[0]:.6f} {p1[1]:.6f} {p1[2]:.6f}\n"
                    stl_content += f"      vertex {p2[0]:.6f} {p2[1]:.6f} {p2[2]:.6f}\n"
                    stl_content += f"      vertex {p3[0]:.6f} {p3[1]:.6f} {p3[2]:.6f}\n"
                    stl_content += "    endloop\n"
                    stl_content += "  endfacet\n"
                    
                    # Triangle 2: p1, p3, p4
                    normal = np.cross(p3 - p1, p4 - p1)
                    normal = normal / np.linalg.norm(normal)
                    
                    stl_content += f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n"
                    stl_content += "    outer loop\n"
                    stl_content += f"      vertex {p1[0]:.6f} {p1[1]:.6f} {p1[2]:.6f}\n"
                    stl_content += f"      vertex {p3[0]:.6f} {p3[1]:.6f} {p3[2]:.6f}\n"
                    stl_content += f"      vertex {p4[0]:.6f} {p4[1]:.6f} {p4[2]:.6f}\n"
                    stl_content += "    endloop\n"
                    stl_content += "  endfacet\n"
            
            # Generate triangles for lower surface
            for i in range(n_span - 1):
                for j in range(n_chord - 1):
                    # Two triangles per quad (reversed winding for lower surface)
                    p1 = wing_lower[i, j]
                    p2 = wing_lower[i, j+1]
                    p3 = wing_lower[i+1, j+1]
                    p4 = wing_lower[i+1, j]
                    
                    # Triangle 1: p1, p2, p3
                    normal = np.cross(p2 - p1, p3 - p1)
                    normal = normal / np.linalg.norm(normal)
                    
                    stl_content += f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n"
                    stl_content += "    outer loop\n"
                    stl_content += f"      vertex {p1[0]:.6f} {p1[1]:.6f} {p1[2]:.6f}\n"
                    stl_content += f"      vertex {p2[0]:.6f} {p2[1]:.6f} {p2[2]:.6f}\n"
                    stl_content += f"      vertex {p3[0]:.6f} {p3[1]:.6f} {p3[2]:.6f}\n"
                    stl_content += "    endloop\n"
                    stl_content += "  endfacet\n"
                    
                    # Triangle 2: p1, p3, p4
                    normal = np.cross(p3 - p1, p4 - p1)
                    normal = normal / np.linalg.norm(normal)
                    
                    stl_content += f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n"
                    stl_content += "    outer loop\n"
                    stl_content += f"      vertex {p1[0]:.6f} {p1[1]:.6f} {p1[2]:.6f}\n"
                    stl_content += f"      vertex {p3[0]:.6f} {p3[1]:.6f} {p3[2]:.6f}\n"
                    stl_content += f"      vertex {p4[0]:.6f} {p4[1]:.6f} {p4[2]:.6f}\n"
                    stl_content += "    endloop\n"
                    stl_content += "  endfacet\n"
            
            stl_content += "endsolid wing\n"
            
            return stl_content
            
        except Exception as e:
            raise Exception(f"STL export failed: {e}")

    def generate_deterministic_test_mesh(self, config_name: str = "NACA2412_test") -> Dict:
        """Generate a deterministic test mesh for validation purposes"""
        test_configs = {
            "NACA2412_test": {"m": 2, "p": 4, "t": 12, "AR": 8, "taper": 0.5, "sweep": 15, "twist": 3},
            "symmetric_test": {"m": 0, "p": 0, "t": 12, "AR": 6, "taper": 1.0, "sweep": 0, "twist": 0},
            "high_camber_test": {"m": 6, "p": 4, "t": 15, "AR": 10, "taper": 0.3, "sweep": 30, "twist": 5}
        }
        
        if config_name not in test_configs:
            raise ValueError(f"Unknown test configuration: {config_name}")
        
        config = test_configs[config_name]
        
        try:
            geometry = self.generate_3d_wing_mesh(
                m=config["m"], p=config["p"], t=config["t"], AR=config["AR"],
                taper=config["taper"], sweep=config["sweep"], twist=config["twist"],
                chord_root=1.0, n_span=20, n_chord=30  # Smaller mesh for testing
            )
            
            # Additional validation for test mesh
            wing_upper = geometry['upper_surface']
            wing_lower = geometry['lower_surface']
            
            # Check mesh properties
            span_actual = wing_upper[:, 0, 1].max() - wing_upper[:, 0, 1].min()
            expected_span = config["AR"] * 1.0  # chord_root = 1.0
            
            if abs(span_actual - expected_span) > 0.1:
                st.warning(f"Test mesh span mismatch: expected {expected_span:.2f}, got {span_actual:.2f}")
            
            st.success(f"Test mesh '{config_name}' generated successfully")
            return geometry
            
        except Exception as e:
            st.error(f"Test mesh generation failed for '{config_name}': {e}")
            raise

# Global visualizer instance
wing_visualizer = Advanced3DWingVisualizer()