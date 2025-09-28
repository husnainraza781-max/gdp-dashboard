"""
Aerodynamic Optimization Module

This module provides advanced optimization capabilities using genetic algorithms,
SciPy optimizers, and hybrid approaches for wing performance optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional, Union
from scipy import optimize
from dataclasses import dataclass
import random
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import warnings

warnings.filterwarnings('ignore')

# Import DEAP for genetic algorithms
try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    # Create dummy objects to avoid unbound variable errors
    base = None
    creator = None
    tools = None
    algorithms = None
    DEAP_AVAILABLE = False
    st.warning("DEAP not available. Using custom genetic algorithm implementation.")


@dataclass
class OptimizationBounds:
    """Define optimization bounds for design variables"""
    m_bounds: Tuple[float, float] = (0, 9)          # NACA maximum camber position
    p_bounds: Tuple[float, float] = (0, 9)          # NACA maximum camber position  
    t_bounds: Tuple[float, float] = (8, 25)         # NACA thickness (%)
    alpha_bounds: Tuple[float, float] = (-5, 15)    # Angle of attack (degrees)
    chord_bounds: Tuple[float, float] = (0.5, 3.0)  # Chord length (m)
    AR_bounds: Tuple[float, float] = (4, 20)        # Aspect ratio
    taper_bounds: Tuple[float, float] = (0.3, 1.0)  # Taper ratio
    sweep_bounds: Tuple[float, float] = (0, 45)     # Sweep angle (degrees)


@dataclass
class OptimizationObjective:
    """Define optimization objectives"""
    name: str
    maximize: bool = True
    weight: float = 1.0
    constraint_type: Optional[str] = None  # 'equality', 'inequality', or None
    constraint_value: Optional[float] = None


class AerodynamicOptimizer:
    """
    Main optimization class supporting multiple algorithms
    """
    
    def __init__(self, analysis_function: Callable):
        self.analysis_function = analysis_function
        self.bounds = OptimizationBounds()
        self.objectives = []
        self.constraints = []
        self.optimization_history = []
        self.best_solution = None
        
    def set_bounds(self, **kwargs):
        """Update optimization bounds"""
        for key, value in kwargs.items():
            if hasattr(self.bounds, key):
                setattr(self.bounds, key, value)
    
    def add_objective(self, name: str, maximize: bool = True, weight: float = 1.0):
        """Add optimization objective"""
        self.objectives.append(OptimizationObjective(name, maximize, weight))
    
    def add_constraint(self, name: str, constraint_type: str, value: float):
        """Add optimization constraint"""
        constraint = OptimizationObjective(name, False, 1.0, constraint_type, value)
        self.constraints.append(constraint)
    
    def evaluate_design(self, design_vars: List[float], problem_type: str = '2d') -> Dict:
        """
        Evaluate a design and return objective values
        """
        try:
            if problem_type == '2d':
                m, p, t, alpha, V, chord = design_vars
                result = self.analysis_function(m, p, t, alpha, V, chord=chord)
                
                # Extract key metrics
                cl = result['aerodynamics']['Cl']
                cd = result['drag']['Cd_total']
                ld_ratio = cl / cd if cd > 0 else 0
                
                return {
                    'Cl': cl,
                    'Cd': cd,
                    'L/D': ld_ratio,
                    'valid': True,
                    'result': result
                }
                
            elif problem_type == '3d':
                m, p, t, alpha, AR, taper, sweep, chord_root, V = design_vars
                
                # Import wing analysis here to avoid circular imports
                from aero import wing_3d_analysis
                result = wing_3d_analysis(m, p, t, alpha, AR, taper, 
                                        sweep_deg=sweep, twist_deg=0, 
                                        chord_root=chord_root, V=V)
                
                # Extract 3D metrics
                aero_3d = result['aerodynamics_3d']
                cl_3d = aero_3d.get('Cl_3d', 0)
                cd_3d = aero_3d.get('Cd_total_3d', 0.1)
                ld_ratio_3d = cl_3d / cd_3d if cd_3d > 0 else 0
                
                return {
                    'Cl_3d': cl_3d,
                    'Cd_3d': cd_3d,
                    'L/D_3d': ld_ratio_3d,
                    'efficiency_factor': aero_3d.get('efficiency_factor', 0.8),
                    'valid': True,
                    'result': result
                }
            else:
                # Unsupported problem type
                return {'valid': False, 'error': f'Unsupported problem type: {problem_type}'}
                
        except Exception as e:
            # Return invalid design
            return {'valid': False, 'error': str(e)}
    
    def objective_function(self, design_vars: List[float], problem_type: str = '2d') -> float:
        """
        Combined objective function for optimization
        """
        eval_result = self.evaluate_design(design_vars, problem_type)
        
        if not eval_result.get('valid', False):
            return 1e6  # Penalty for invalid designs
        
        # Calculate weighted objective
        total_objective = 0
        
        for obj in self.objectives:
            if obj.name in eval_result:
                value = eval_result[obj.name]
                # Maximize objectives are negated for minimization algorithms
                if obj.maximize:
                    total_objective -= obj.weight * value
                else:
                    total_objective += obj.weight * value
        
        # Add constraint penalties
        for constraint in self.constraints:
            if constraint.name in eval_result:
                value = eval_result[constraint.name]
                if constraint.constraint_type == 'inequality':
                    # Constraint: value <= constraint_value
                    if value > constraint.constraint_value:
                        total_objective += 1000 * (value - constraint.constraint_value)**2
                elif constraint.constraint_type == 'equality':
                    # Constraint: value == constraint_value
                    total_objective += 1000 * (value - constraint.constraint_value)**2
        
        return total_objective
    
    def optimize_scipy(self, problem_type: str = '2d', method: str = 'L-BFGS-B', 
                      max_iter: int = 100) -> Dict:
        """
        Optimize using SciPy optimizers
        """
        # Define bounds based on problem type
        if problem_type == '2d':
            bounds = [
                self.bounds.m_bounds, self.bounds.p_bounds, self.bounds.t_bounds,
                self.bounds.alpha_bounds, (50, 200), self.bounds.chord_bounds
            ]
            # Initial guess
            x0 = [2, 4, 12, 5, 100, 1.0]
        elif problem_type == '3d':
            bounds = [
                self.bounds.m_bounds, self.bounds.p_bounds, self.bounds.t_bounds,
                self.bounds.alpha_bounds, self.bounds.AR_bounds, self.bounds.taper_bounds,
                self.bounds.sweep_bounds, self.bounds.chord_bounds, (50, 200)
            ]
            # Initial guess
            x0 = [2, 4, 12, 5, 8, 0.6, 10, 1.5, 100]
        
        # Optimization
        result = optimize.minimize(
            fun=lambda x: self.objective_function(x, problem_type),
            x0=x0,
            method=method,
            bounds=bounds,
            options={'maxiter': max_iter}
        )
        
        # Evaluate best solution
        best_eval = self.evaluate_design(result.x, problem_type)
        
        return {
            'success': result.success,
            'message': result.message,
            'best_design': result.x,
            'best_objective': result.fun,
            'best_evaluation': best_eval,
            'iterations': result.nit,
            'method': method
        }
    
    def optimize_genetic(self, problem_type: str = '2d', population_size: int = 50,
                        generations: int = 50, mutation_rate: float = 0.1) -> Dict:
        """
        Optimize using genetic algorithm
        """
        if DEAP_AVAILABLE:
            return self._optimize_genetic_deap(problem_type, population_size, 
                                             generations, mutation_rate)
        else:
            return self._optimize_genetic_custom(problem_type, population_size,
                                                generations, mutation_rate)
    
    def _optimize_genetic_deap(self, problem_type: str, population_size: int,
                              generations: int, mutation_rate: float) -> Dict:
        """
        Genetic optimization using DEAP library
        """
        # Setup DEAP
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        toolbox = base.Toolbox()
        
        # Define bounds
        if problem_type == '2d':
            bounds = [
                self.bounds.m_bounds, self.bounds.p_bounds, self.bounds.t_bounds,
                self.bounds.alpha_bounds, (50, 200), self.bounds.chord_bounds
            ]
            n_vars = 6
        else:
            bounds = [
                self.bounds.m_bounds, self.bounds.p_bounds, self.bounds.t_bounds,
                self.bounds.alpha_bounds, self.bounds.AR_bounds, self.bounds.taper_bounds,
                self.bounds.sweep_bounds, self.bounds.chord_bounds, (50, 200)
            ]
            n_vars = 9
        
        # Register functions
        for i in range(n_vars):
            toolbox.register(f"attr_{i}", random.uniform, bounds[i][0], bounds[i][1])
        
        toolbox.register("individual", tools.initCycle, creator.Individual,
                        [getattr(toolbox, f"attr_{i}") for i in range(n_vars)], n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        toolbox.register("evaluate", lambda x: (self.objective_function(x, problem_type),))
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=mutation_rate)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Create population and evolve
        population = toolbox.population(n=population_size)
        algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=mutation_rate,
                           ngen=generations, verbose=False)
        
        # Find best individual
        best_ind = tools.selBest(population, 1)[0]
        best_eval = self.evaluate_design(best_ind, problem_type)
        
        return {
            'success': True,
            'best_design': list(best_ind),
            'best_objective': best_ind.fitness.values[0],
            'best_evaluation': best_eval,
            'generations': generations,
            'method': 'genetic_deap'
        }
    
    def _optimize_genetic_custom(self, problem_type: str, population_size: int,
                                generations: int, mutation_rate: float) -> Dict:
        """
        Custom genetic algorithm implementation
        """
        # Define bounds
        if problem_type == '2d':
            bounds = [
                self.bounds.m_bounds, self.bounds.p_bounds, self.bounds.t_bounds,
                self.bounds.alpha_bounds, (50, 200), self.bounds.chord_bounds
            ]
            n_vars = 6
        else:
            bounds = [
                self.bounds.m_bounds, self.bounds.p_bounds, self.bounds.t_bounds,
                self.bounds.alpha_bounds, self.bounds.AR_bounds, self.bounds.taper_bounds,
                self.bounds.sweep_bounds, self.bounds.chord_bounds, (50, 200)
            ]
            n_vars = 9
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = [random.uniform(bound[0], bound[1]) for bound in bounds]
            population.append(individual)
        
        best_fitness = float('inf')
        best_individual = None
        fitness_history = []
        
        for generation in range(generations):
            # Evaluate population
            fitness_scores = []
            for individual in population:
                fitness = self.objective_function(individual, problem_type)
                fitness_scores.append(fitness)
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
            
            fitness_history.append(best_fitness)
            
            # Selection (tournament)
            new_population = []
            for _ in range(population_size):
                # Tournament selection
                tournament_size = min(3, population_size)
                tournament_indices = random.sample(range(population_size), tournament_size)
                winner_idx = min(tournament_indices, key=lambda i: fitness_scores[i])
                new_population.append(population[winner_idx].copy())
            
            # Crossover and mutation
            for i in range(0, population_size - 1, 2):
                if random.random() < 0.7:  # Crossover probability
                    # Two-point crossover
                    if len(new_population[i]) > 2:
                        point1 = random.randint(1, len(new_population[i]) - 2)
                        point2 = random.randint(point1, len(new_population[i]) - 1)
                        
                        temp = new_population[i][point1:point2]
                        new_population[i][point1:point2] = new_population[i+1][point1:point2]
                        new_population[i+1][point1:point2] = temp
                
                # Mutation
                for j in range(2):
                    if i + j < population_size:
                        for k in range(n_vars):
                            if random.random() < mutation_rate:
                                mutation_strength = 0.1 * (bounds[k][1] - bounds[k][0])
                                new_population[i+j][k] += random.gauss(0, mutation_strength)
                                # Ensure bounds
                                new_population[i+j][k] = max(bounds[k][0], 
                                                           min(bounds[k][1], new_population[i+j][k]))
            
            population = new_population
        
        best_eval = self.evaluate_design(best_individual, problem_type)
        
        return {
            'success': True,
            'best_design': best_individual,
            'best_objective': best_fitness,
            'best_evaluation': best_eval,
            'generations': generations,
            'method': 'genetic_custom',
            'fitness_history': fitness_history
        }
    
    def optimize_hybrid(self, problem_type: str = '2d', ga_generations: int = 30,
                       scipy_method: str = 'L-BFGS-B') -> Dict:
        """
        Hybrid optimization: GA for global search + SciPy for local refinement
        """
        # Stage 1: Genetic Algorithm
        ga_result = self.optimize_genetic(problem_type, population_size=30,
                                        generations=ga_generations)
        
        if not ga_result['success']:
            return ga_result
        
        # Stage 2: Local refinement with SciPy
        if problem_type == '2d':
            bounds = [
                self.bounds.m_bounds, self.bounds.p_bounds, self.bounds.t_bounds,
                self.bounds.alpha_bounds, (50, 200), self.bounds.chord_bounds
            ]
        else:
            bounds = [
                self.bounds.m_bounds, self.bounds.p_bounds, self.bounds.t_bounds,
                self.bounds.alpha_bounds, self.bounds.AR_bounds, self.bounds.taper_bounds,
                self.bounds.sweep_bounds, self.bounds.chord_bounds, (50, 200)
            ]
        
        scipy_result = optimize.minimize(
            fun=lambda x: self.objective_function(x, problem_type),
            x0=ga_result['best_design'],
            method=scipy_method,
            bounds=bounds,
            options={'maxiter': 50}
        )
        
        # Use better of GA or SciPy result
        if scipy_result.success and scipy_result.fun < ga_result['best_objective']:
            best_eval = self.evaluate_design(scipy_result.x, problem_type)
            return {
                'success': True,
                'best_design': scipy_result.x,
                'best_objective': scipy_result.fun,
                'best_evaluation': best_eval,
                'method': 'hybrid',
                'ga_result': ga_result,
                'scipy_improvement': True
            }
        else:
            return {
                'success': True,
                'best_design': ga_result['best_design'],
                'best_objective': ga_result['best_objective'],
                'best_evaluation': ga_result['best_evaluation'],
                'method': 'hybrid',
                'ga_result': ga_result,
                'scipy_improvement': False
            }
    
    def plot_optimization_results(self, results: Dict, problem_type: str = '2d') -> go.Figure:
        """
        Create visualization of optimization results
        """
        if not results.get('success', False):
            return go.Figure().add_annotation(text="Optimization failed", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        best_eval = results['best_evaluation']
        best_design = results['best_design']
        
        if problem_type == '2d':
            var_names = ['m', 'p', 't (%)', 'α (°)', 'V (m/s)', 'chord (m)']
            obj_names = ['Cl', 'Cd', 'L/D']
        else:
            var_names = ['m', 'p', 't (%)', 'α (°)', 'AR', 'taper', 'sweep (°)', 'chord (m)', 'V (m/s)']
            obj_names = ['Cl_3d', 'Cd_3d', 'L/D_3d', 'efficiency_factor']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Design Variables', 'Objective Values', 
                          'Optimization Progress', 'Performance Summary'],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "table"}]]
        )
        
        # Design variables bar chart
        fig.add_trace(
            go.Bar(x=var_names, y=best_design, name='Design Variables'),
            row=1, col=1
        )
        
        # Objective values bar chart
        obj_values = [best_eval.get(name, 0) for name in obj_names if name in best_eval]
        obj_labels = [name for name in obj_names if name in best_eval]
        
        fig.add_trace(
            go.Bar(x=obj_labels, y=obj_values, name='Objectives'),
            row=1, col=2
        )
        
        # Optimization progress (if available)
        if 'fitness_history' in results:
            generations = list(range(len(results['fitness_history'])))
            fig.add_trace(
                go.Scatter(x=generations, y=results['fitness_history'], 
                          mode='lines+markers', name='Best Fitness'),
                row=2, col=1
            )
        
        # Performance summary table
        table_data = []
        for name in obj_labels:
            if name in best_eval:
                table_data.append([name, f"{best_eval[name]:.4f}"])
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=list(zip(*table_data)) if table_data else [[], []])
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"Optimization Results ({results['method']})",
            showlegend=False
        )
        
        return fig


def multi_objective_pareto_optimization(optimizer: AerodynamicOptimizer, 
                                      problem_type: str = '2d',
                                      n_points: int = 20) -> Dict:
    """
    Perform multi-objective Pareto optimization
    """
    pareto_solutions = []
    
    # Generate multiple solutions by varying objective weights
    for i in range(n_points):
        # Vary weights between objectives
        w1 = i / (n_points - 1)
        w2 = 1 - w1
        
        # Clear existing objectives and add weighted ones
        optimizer.objectives = []
        optimizer.add_objective('L/D' if problem_type == '2d' else 'L/D_3d', True, w1)
        optimizer.add_objective('Cl' if problem_type == '2d' else 'Cl_3d', True, w2)
        
        # Optimize
        result = optimizer.optimize_scipy(problem_type, method='L-BFGS-B', max_iter=50)
        
        if result['success']:
            pareto_solutions.append({
                'design': result['best_design'],
                'evaluation': result['best_evaluation'],
                'weights': [w1, w2]
            })
    
    return {
        'pareto_solutions': pareto_solutions,
        'n_solutions': len(pareto_solutions)
    }


# Example usage and testing
if __name__ == "__main__":
    # This would be used with actual aerodynamic analysis function
    def dummy_analysis(m, p, t, alpha, V, chord=1.0):
        """Dummy analysis for testing"""
        cl = 0.1 * alpha + 0.001 * t
        cd = 0.01 + 0.0001 * alpha**2
        return {
            'Cl_corrected': cl,
            'Cd_total': cd
        }
    
    # Create optimizer
    optimizer = AerodynamicOptimizer(dummy_analysis)
    optimizer.add_objective('L/D', maximize=True, weight=1.0)
    
    # Test optimization
    result = optimizer.optimize_scipy('2d')
    print("Optimization result:", result)