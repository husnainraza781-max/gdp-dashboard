"""
AI-Powered Design Optimization Module
Provides intelligent design recommendations using machine learning models trained on experimental data
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
from typing import Dict, List, Tuple, Optional
import pickle
import os
from experimental_data import experimental_db

class AIDesignOptimizer:
    """AI-powered aerodynamic design optimizer using machine learning"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.model_performance = {}
        
    def prepare_training_data(self) -> pd.DataFrame:
        """
        Prepare training data from experimental datasets
        """
        # Get all experimental data
        airfoil_data = experimental_db.datasets['naca_airfoils']
        wing_3d_data = experimental_db.datasets['naca_wings_3d']
        
        # Prepare 2D airfoil training data
        airfoil_features = []
        
        for _, row in airfoil_data.iterrows():
            # Extract NACA parameters from airfoil name
            naca = row['airfoil']
            if len(naca) == 4:
                m = int(naca[0])
                p = int(naca[1]) 
                t = int(naca[2:4])
                
                features = {
                    'm': m,
                    'p': p, 
                    't': t,
                    'alpha': row['alpha'],
                    'Re': row['Re'],
                    'Mach': row['Mach'],
                    'Cl': row['Cl'],
                    'Cd': row['Cd'],
                    'LD': row['LD'],
                    'type': '2d'
                }
                airfoil_features.append(features)
        
        # Prepare 3D wing training data
        for _, row in wing_3d_data.iterrows():
            # Extract NACA parameters
            naca = row['airfoil']
            if len(naca) == 4:
                m = int(naca[0])
                p = int(naca[1])
                t = int(naca[2:4])
                
                features = {
                    'm': m,
                    'p': p,
                    't': t,
                    'alpha': row['alpha'],
                    'AR': row['AR'],
                    'taper': row['taper'],
                    'sweep': row['sweep'],
                    'Re': row['Re'],
                    'Mach': row['Mach'],
                    'CL': row['CL'],
                    'CD': row['CD'],
                    'LD': row['LD'],
                    'type': '3d'
                }
                airfoil_features.append(features)
        
        return pd.DataFrame(airfoil_features)
    
    def train_models(self) -> Dict:
        """
        Train AI models on experimental data
        """
        training_data = self.prepare_training_data()
        
        if len(training_data) < 10:
            raise ValueError("Insufficient training data")
        
        results = {}
        
        # Train 2D airfoil models
        airfoil_2d_data = training_data[training_data['type'] == '2d'].copy()
        if len(airfoil_2d_data) > 5:
            
            # Features for 2D prediction
            feature_cols_2d = ['m', 'p', 't', 'alpha', 'Re', 'Mach']
            X_2d = airfoil_2d_data[feature_cols_2d]
            
            # Train Cl prediction model
            y_cl = airfoil_2d_data['Cl']
            X_train, X_test, y_train, y_test = train_test_split(X_2d, y_cl, test_size=0.2, random_state=42)
            
            # Scale features
            scaler_2d = StandardScaler()
            X_train_scaled = scaler_2d.fit_transform(X_train)
            X_test_scaled = scaler_2d.transform(X_test)
            
            # Train Random Forest model
            rf_cl = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_cl.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = rf_cl.predict(X_test_scaled)
            r2_cl = r2_score(y_test, y_pred)
            rmse_cl = np.sqrt(mean_squared_error(y_test, y_pred))
            
            self.models['2d_cl'] = rf_cl
            self.scalers['2d'] = scaler_2d
            results['2d_cl'] = {'r2': r2_cl, 'rmse': rmse_cl}
            
            # Train Cd prediction model
            y_cd = airfoil_2d_data['Cd']
            _, _, y_train_cd, y_test_cd = train_test_split(X_2d, y_cd, test_size=0.2, random_state=42)
            
            rf_cd = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_cd.fit(X_train_scaled, y_train_cd)
            
            y_pred_cd = rf_cd.predict(X_test_scaled)
            r2_cd = r2_score(y_test_cd, y_pred_cd)
            rmse_cd = np.sqrt(mean_squared_error(y_test_cd, y_pred_cd))
            
            self.models['2d_cd'] = rf_cd
            results['2d_cd'] = {'r2': r2_cd, 'rmse': rmse_cd}
        
        # Train 3D wing models
        wing_3d_data = training_data[training_data['type'] == '3d'].copy()
        if len(wing_3d_data) > 5:
            
            feature_cols_3d = ['m', 'p', 't', 'alpha', 'AR', 'taper', 'sweep', 'Re', 'Mach']
            X_3d = wing_3d_data[feature_cols_3d]
            
            # Train CL prediction model for 3D
            y_cl_3d = wing_3d_data['CL']
            X_train_3d, X_test_3d, y_train_3d, y_test_3d = train_test_split(X_3d, y_cl_3d, test_size=0.2, random_state=42)
            
            scaler_3d = StandardScaler()
            X_train_3d_scaled = scaler_3d.fit_transform(X_train_3d)
            X_test_3d_scaled = scaler_3d.transform(X_test_3d)
            
            rf_cl_3d = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_cl_3d.fit(X_train_3d_scaled, y_train_3d)
            
            y_pred_3d = rf_cl_3d.predict(X_test_3d_scaled)
            r2_cl_3d = r2_score(y_test_3d, y_pred_3d)
            rmse_cl_3d = np.sqrt(mean_squared_error(y_test_3d, y_pred_3d))
            
            self.models['3d_cl'] = rf_cl_3d
            self.scalers['3d'] = scaler_3d
            results['3d_cl'] = {'r2': r2_cl_3d, 'rmse': rmse_cl_3d}
        
        self.model_performance = results
        self.is_trained = True
        
        return results
    
    def predict_performance(self, design_params: Dict, design_type: str = '2d') -> Dict:
        """
        Predict aerodynamic performance using trained AI models
        """
        if not self.is_trained:
            raise ValueError("Models not trained. Call train_models() first.")
        
        if design_type == '2d':
            if '2d_cl' not in self.models:
                raise ValueError("2D models not available")
            
            # Prepare features
            features = np.array([[
                design_params.get('m', 0),
                design_params.get('p', 4),
                design_params.get('t', 12),
                design_params.get('alpha', 5),
                design_params.get('Re', 3e6),
                design_params.get('Mach', 0.15)
            ]])
            
            # Scale features
            features_scaled = self.scalers['2d'].transform(features)
            
            # Predict
            cl_pred = self.models['2d_cl'].predict(features_scaled)[0]
            cd_pred = self.models['2d_cd'].predict(features_scaled)[0] if '2d_cd' in self.models else 0.01
            
            return {
                'Cl_ai': cl_pred,
                'Cd_ai': cd_pred,
                'LD_ai': cl_pred / cd_pred if cd_pred > 0 else 0,
                'confidence_cl': self.model_performance.get('2d_cl', {}).get('r2', 0),
                'confidence_cd': self.model_performance.get('2d_cd', {}).get('r2', 0)
            }
        
        elif design_type == '3d':
            if '3d_cl' not in self.models:
                raise ValueError("3D models not available")
            
            features = np.array([[
                design_params.get('m', 0),
                design_params.get('p', 4), 
                design_params.get('t', 12),
                design_params.get('alpha', 5),
                design_params.get('AR', 8),
                design_params.get('taper', 1.0),
                design_params.get('sweep', 0),
                design_params.get('Re', 3e6),
                design_params.get('Mach', 0.15)
            ]])
            
            features_scaled = self.scalers['3d'].transform(features)
            cl_pred = self.models['3d_cl'].predict(features_scaled)[0]
            
            return {
                'CL_ai': cl_pred,
                'confidence': self.model_performance.get('3d_cl', {}).get('r2', 0)
            }
        else:
            raise ValueError(f"Unknown design_type: {design_type}. Must be '2d' or '3d'.")
    
    def generate_design_recommendations(self, target_performance: Dict, design_type: str = '2d') -> List[Dict]:
        """
        Generate AI-powered design recommendations for target performance
        """
        if not self.is_trained:
            self.train_models()
        
        recommendations = []
        
        # Design space exploration
        if design_type == '2d':
            # Explore NACA 4-digit parameter space
            m_range = range(0, 7)
            p_range = range(2, 9) 
            t_range = range(8, 21)
            alpha_range = np.linspace(0, 12, 7)
            
            best_designs = []
            
            for m in m_range:
                for p in p_range:
                    for t in t_range:
                        for alpha in alpha_range:
                            design = {
                                'm': m, 'p': p, 't': t, 'alpha': alpha,
                                'Re': 3e6, 'Mach': 0.15
                            }
                            
                            try:
                                pred = self.predict_performance(design, '2d')
                                
                                # Score based on target performance
                                score = 0
                                if 'target_cl' in target_performance:
                                    score += 1 / (1 + abs(pred['Cl_ai'] - target_performance['target_cl']))
                                if 'target_ld' in target_performance:
                                    score += 1 / (1 + abs(pred['LD_ai'] - target_performance['target_ld']))
                                
                                design_result = {**design, **pred, 'score': score}
                                best_designs.append(design_result)
                                
                            except:
                                continue
            
            # Sort by score and return top recommendations
            best_designs.sort(key=lambda x: x['score'], reverse=True)
            recommendations = best_designs[:5]
        
        return recommendations
    
    def save_models(self, filepath: str = 'ai_models.pkl'):
        """Save trained models to file"""
        if self.is_trained:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'performance': self.model_performance
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
    
    def load_models(self, filepath: str = 'ai_models.pkl') -> bool:
        """Load trained models from file"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                self.models = model_data['models']
                self.scalers = model_data['scalers']
                self.model_performance = model_data['performance']
                self.is_trained = True
                return True
            except:
                return False
        return False


# Global AI optimizer instance
ai_optimizer = AIDesignOptimizer()