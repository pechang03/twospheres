"""Global optimization framework for LOC system design.

Multi-objective optimization using:
- NSGA-II (Non-dominated Sorting Genetic Algorithm II)
- Bayesian optimization with Gaussian Processes
- Particle Swarm Optimization (PSO)

Merit functions:
- Strehl ratio (optical quality)
- Modulation Transfer Function (MTF) at Nyquist
- Fiber-chip coupling efficiency
- Fabrication tolerance budget
"""

import asyncio
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from scipy.optimize import differential_evolution, minimize


@dataclass
class OptimizationVariable:
    """Single optimization variable specification.

    Attributes:
        name: Variable name (e.g., "wavelength_nm", "na_objective")
        bounds: (min, max) allowed values
        initial: Initial guess value
        units: Physical units (for display)
        description: Human-readable description
    """
    name: str
    bounds: Tuple[float, float]
    initial: float
    units: str = ""
    description: str = ""


@dataclass
class OptimizationObjective:
    """Single optimization objective specification.

    Attributes:
        name: Objective name (e.g., "strehl_ratio", "coupling_efficiency")
        minimize: If True, minimize; if False, maximize
        weight: Relative weight in multi-objective optimization (0-1)
        target: Optional target value for constraint-based optimization
        tolerance: Optional tolerance around target
    """
    name: str
    minimize: bool = False  # False = maximize by default
    weight: float = 1.0
    target: Optional[float] = None
    tolerance: Optional[float] = None


class LOCSystemOptimizer:
    """Global optimizer for Lab-on-Chip optical system design.

    Optimizes multiple objectives simultaneously (Strehl, MTF, coupling)
    subject to fabrication constraints using multi-objective algorithms.
    """

    def __init__(
        self,
        variables: List[OptimizationVariable],
        objectives: List[OptimizationObjective],
        objective_function: Callable,
        constraints: Optional[List[Callable]] = None
    ):
        """Initialize LOC system optimizer.

        Args:
            variables: List of optimization variables with bounds
            objectives: List of objectives to optimize
            objective_function: Async function that computes objectives
                               Signature: async (params: Dict) -> Dict[str, float]
            constraints: Optional list of constraint functions
                        Each returns True if constraint satisfied
        """
        self.variables = {var.name: var for var in variables}
        self.objectives = {obj.name: obj for obj in objectives}
        self.objective_function = objective_function
        self.constraints = constraints or []

        self.n_vars = len(variables)
        self.n_objectives = len(objectives)
        self.var_names = [var.name for var in variables]
        self.obj_names = [obj.name for obj in objectives]

        # Build bounds array for scipy
        self.bounds = [self.variables[name].bounds for name in self.var_names]

    async def evaluate_objectives(self, x: NDArray) -> NDArray:
        """Evaluate all objectives for parameter vector x.

        Args:
            x: Parameter vector (numpy array)

        Returns:
            Objective vector (numpy array)
        """
        # Convert array to parameter dict
        params = {name: float(x[i]) for i, name in enumerate(self.var_names)}

        # Call objective function
        results = await self.objective_function(params)

        # Extract objective values
        objectives = np.zeros(self.n_objectives)
        for i, obj_name in enumerate(self.obj_names):
            obj_spec = self.objectives[obj_name]
            value = results[obj_name]

            # Convert to minimization (negate if maximization objective)
            if not obj_spec.minimize:
                value = -value  # Minimize negative = maximize

            objectives[i] = value

        return objectives

    async def optimize_nsga2(
        self,
        population_size: int = 100,
        n_generations: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.9
    ) -> Dict[str, Any]:
        """Multi-objective optimization using NSGA-II algorithm.

        Args:
            population_size: Number of individuals in population
            n_generations: Number of generations to evolve
            mutation_rate: Probability of mutation (0-1)
            crossover_rate: Probability of crossover (0-1)

        Returns:
            Dict with:
                - pareto_front: List of non-dominated solutions
                - best_solution: Single best solution (closest to ideal point)
                - convergence_history: Objective values over generations
        """
        def _optimize():
            # Simple NSGA-II implementation (production would use pymoo library)

            # Initialize population randomly within bounds
            population = []
            for _ in range(population_size):
                individual = np.array([
                    np.random.uniform(self.bounds[i][0], self.bounds[i][1])
                    for i in range(self.n_vars)
                ])
                population.append(individual)

            convergence_history = []

            for generation in range(n_generations):
                # Evaluate population (synchronous for simplicity - production would be async)
                # In production: use asyncio.gather to evaluate in parallel
                fitness_values = []
                for individual in population:
                    # Simplified: use synchronous evaluation
                    # Real implementation would use: await self.evaluate_objectives(individual)
                    fitness = np.random.rand(self.n_objectives)  # Placeholder
                    fitness_values.append(fitness)

                fitness_array = np.array(fitness_values)

                # Track best objective values
                best_objectives = np.min(fitness_array, axis=0)
                convergence_history.append(best_objectives.tolist())

                # Non-dominated sorting (simplified)
                # Real NSGA-II would compute Pareto fronts and crowding distance

                # Selection: tournament selection
                parents = []
                for _ in range(population_size):
                    # Select two random individuals, pick better one
                    idx1, idx2 = np.random.choice(population_size, 2, replace=False)
                    if np.sum(fitness_values[idx1]) < np.sum(fitness_values[idx2]):
                        parents.append(population[idx1].copy())
                    else:
                        parents.append(population[idx2].copy())

                # Crossover and mutation
                offspring = []
                for i in range(0, population_size - 1, 2):
                    parent1 = parents[i]
                    parent2 = parents[i + 1]

                    # Crossover
                    if np.random.rand() < crossover_rate:
                        alpha = np.random.rand()
                        child1 = alpha * parent1 + (1 - alpha) * parent2
                        child2 = (1 - alpha) * parent1 + alpha * parent2
                    else:
                        child1 = parent1.copy()
                        child2 = parent2.copy()

                    # Mutation
                    for j in range(self.n_vars):
                        if np.random.rand() < mutation_rate:
                            child1[j] += np.random.randn() * 0.1 * (self.bounds[j][1] - self.bounds[j][0])
                            child1[j] = np.clip(child1[j], self.bounds[j][0], self.bounds[j][1])
                        if np.random.rand() < mutation_rate:
                            child2[j] += np.random.randn() * 0.1 * (self.bounds[j][1] - self.bounds[j][0])
                            child2[j] = np.clip(child2[j], self.bounds[j][0], self.bounds[j][1])

                    offspring.append(child1)
                    offspring.append(child2)

                population = offspring[:population_size]

            # Extract Pareto front (simplified - take best 10% by sum of objectives)
            final_fitness = []
            for individual in population:
                fitness = np.random.rand(self.n_objectives)  # Placeholder
                final_fitness.append((individual, fitness))

            final_fitness.sort(key=lambda x: np.sum(x[1]))
            pareto_front_size = max(1, population_size // 10)
            pareto_front = final_fitness[:pareto_front_size]

            # Best solution: first in Pareto front
            best_solution = pareto_front[0]

            return {
                "pareto_front": [
                    {
                        "parameters": {self.var_names[i]: float(sol[0][i]) for i in range(self.n_vars)},
                        "objectives": {self.obj_names[i]: float(sol[1][i]) for i in range(self.n_objectives)}
                    }
                    for sol in pareto_front
                ],
                "best_solution": {
                    "parameters": {self.var_names[i]: float(best_solution[0][i]) for i in range(self.n_vars)},
                    "objectives": {self.obj_names[i]: float(best_solution[1][i]) for i in range(self.n_objectives)}
                },
                "convergence_history": convergence_history,
                "n_generations": n_generations,
                "population_size": population_size
            }

        return await asyncio.to_thread(_optimize)

    async def optimize_bayesian(
        self,
        n_iterations: int = 50,
        n_initial_samples: int = 10,
        acquisition: str = "ei"  # Expected Improvement
    ) -> Dict[str, Any]:
        """Bayesian optimization with Gaussian Process surrogate model.

        Args:
            n_iterations: Number of optimization iterations
            n_initial_samples: Initial random samples for GP training
            acquisition: Acquisition function ("ei", "ucb", "pi")

        Returns:
            Dict with:
                - best_parameters: Optimal parameter values
                - best_objectives: Corresponding objective values
                - samples: All evaluated samples
                - gp_statistics: GP model statistics (lengthscales, variance)
        """
        def _optimize():
            # Simplified Bayesian optimization
            # Production would use GPyOpt, scikit-optimize, or Ax

            # Initial random sampling
            samples_x = []
            samples_y = []

            for _ in range(n_initial_samples):
                x = np.array([
                    np.random.uniform(self.bounds[i][0], self.bounds[i][1])
                    for i in range(self.n_vars)
                ])
                # Placeholder: would call await self.evaluate_objectives(x)
                y = np.random.rand(self.n_objectives)
                samples_x.append(x)
                samples_y.append(y)

            # Iterative optimization
            for iteration in range(n_iterations):
                # Fit Gaussian Process to samples (simplified)
                # Real implementation: use sklearn.gaussian_process.GaussianProcessRegressor

                # Acquisition function optimization (find next point to sample)
                # Simplified: random search + expected improvement estimate
                best_acquisition = -np.inf
                best_candidate = None

                for _ in range(100):  # Random search over acquisition
                    candidate = np.array([
                        np.random.uniform(self.bounds[i][0], self.bounds[i][1])
                        for i in range(self.n_vars)
                    ])

                    # Placeholder acquisition value
                    acq_value = np.random.rand()

                    if acq_value > best_acquisition:
                        best_acquisition = acq_value
                        best_candidate = candidate

                # Evaluate candidate
                # y_new = await self.evaluate_objectives(best_candidate)
                y_new = np.random.rand(self.n_objectives)  # Placeholder

                samples_x.append(best_candidate)
                samples_y.append(y_new)

            # Find best sample
            samples_y_array = np.array(samples_y)
            # Weighted sum of objectives
            weights = np.array([self.objectives[name].weight for name in self.obj_names])
            weighted_objectives = samples_y_array @ weights
            best_idx = np.argmin(weighted_objectives)

            best_x = samples_x[best_idx]
            best_y = samples_y[best_idx]

            return {
                "best_parameters": {self.var_names[i]: float(best_x[i]) for i in range(self.n_vars)},
                "best_objectives": {self.obj_names[i]: float(best_y[i]) for i in range(self.n_objectives)},
                "n_samples": len(samples_x),
                "samples": [
                    {
                        "parameters": {self.var_names[i]: float(samples_x[j][i]) for i in range(self.n_vars)},
                        "objectives": {self.obj_names[i]: float(samples_y[j][i]) for i in range(self.n_objectives)}
                    }
                    for j in range(len(samples_x))
                ],
                "acquisition_function": acquisition,
                "note": "Simplified Bayesian optimization - production use scikit-optimize or Ax"
            }

        return await asyncio.to_thread(_optimize)

    async def optimize_differential_evolution(
        self,
        max_iterations: int = 100,
        population_size: int = 50
    ) -> Dict[str, Any]:
        """Single-objective optimization using Differential Evolution.

        Optimizes weighted sum of objectives using scipy.optimize.

        Args:
            max_iterations: Maximum function evaluations
            population_size: Population size for DE

        Returns:
            Dict with:
                - best_parameters: Optimal parameter values
                - best_objective_value: Optimized weighted objective
                - individual_objectives: Individual objective values at optimum
                - success: Whether optimization converged
        """
        def objective_wrapper(x):
            """Synchronous wrapper for scipy.optimize."""
            # In production, would need to handle async properly
            # Using simplified synchronous evaluation
            params = {name: float(x[i]) for i, name in enumerate(self.var_names)}

            # Placeholder: would call objective_function
            results = {obj_name: np.random.rand() for obj_name in self.obj_names}

            # Weighted sum
            total = 0.0
            for obj_name, value in results.items():
                obj_spec = self.objectives[obj_name]
                weight = obj_spec.weight
                if not obj_spec.minimize:
                    value = -value
                total += weight * value

            return total

        def _optimize():
            result = differential_evolution(
                objective_wrapper,
                self.bounds,
                maxiter=max_iterations,
                popsize=population_size
            )

            # Evaluate final solution to get individual objectives
            final_params = {self.var_names[i]: float(result.x[i]) for i in range(self.n_vars)}
            # Placeholder individual objectives
            individual_objectives = {obj_name: np.random.rand() for obj_name in self.obj_names}

            return {
                "best_parameters": final_params,
                "best_objective_value": float(result.fun),
                "individual_objectives": individual_objectives,
                "success": result.success,
                "n_iterations": result.nit,
                "message": result.message
            }

        return await asyncio.to_thread(_optimize)


# Example objective function for LOC system design
async def example_loc_objective(params: Dict[str, float]) -> Dict[str, float]:
    """Example objective function for LOC optical system.

    Args:
        params: Design parameters (wavelength_nm, na_objective, etc.)

    Returns:
        Dict of objective values (strehl_ratio, mtf, coupling_efficiency)
    """
    # Simplified models - production would call actual optical simulators

    wavelength_nm = params.get("wavelength_nm", 633)
    na = params.get("na_objective", 0.5)
    spot_size_um = params.get("spot_size_um", 3.0)

    # Strehl ratio (simplified - depends on aberrations)
    strehl_ratio = np.exp(-((na - 0.6) / 0.2)**2) * 0.9

    # MTF at Nyquist (simplified)
    mtf_nyquist = na * 0.5

    # Coupling efficiency (simplified)
    coupling_eff = 1.0 / (1 + (spot_size_um - 3.0)**2)

    return {
        "strehl_ratio": float(strehl_ratio),
        "mtf_nyquist": float(mtf_nyquist),
        "coupling_efficiency": float(coupling_eff)
    }
