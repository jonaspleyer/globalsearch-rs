/// Six-Hump Camel Function with COBYLA (Derivative-Free Optimization)
/// 
/// This example demonstrates using COBYLA, a derivative-free local solver, which is particularly 
/// useful when gradients are unavailable, expensive to compute, or unreliable.
///
/// The Six-Hump Camel function is defined as follows:
///
/// $f(x) = (4 - 2.1 x_1^2 + x_1^4 / 3) x_1^2 + x_1 x_2 + (-4 + 4 x_2^2) x_2^2$
///
/// The function is defined on the domain $x_1 \in [-3, 3]$ and $x_2 \in [-2, 2]$.
/// The function has two global minima at $f(0.0898, -0.7126) = -1.0316$ and $f(-0.0898, 0.7126) = -1.0316$.
/// The function is continuous, differentiable and non-convex.
///
/// ## When to Use COBYLA:
/// - When gradient computation is unavailable or unreliable
/// - For noisy objective functions
/// - When the objective function is not differentiable everywhere
/// - For problems with constraints (COBYLA supports general constraints)
/// - When function evaluations are expensive but derivatives are even more expensive
///
/// ## COBYLA vs Gradient-Based Methods:
/// - **Pros**: No derivatives needed, handles constraints naturally, robust to noise
/// - **Cons**: Generally slower convergence, more function evaluations required
///
/// References:
///
/// Molga, M., & Smutnicki, C. Test functions for optimization needs (April 3, 2005), pp. 27-28. Retrieved January 2025, from https://robertmarks.org/Classes/ENGR5358/Papers/functions.pdf
use globalsearch::local_solver::builders::COBYLABuilder;
use globalsearch::problem::Problem;
use globalsearch::{
    oqnlp::OQNLP,
    types::{EvaluationError, LocalSolverType, OQNLPParams, SolutionSet},
};
use ndarray::{array, Array1, Array2};

#[derive(Debug, Clone)]
pub struct SixHumpCamel;

impl Problem for SixHumpCamel {
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
        Ok(
            (4.0 - 2.1 * x[0].powi(2) + x[0].powi(4) / 3.0) * x[0].powi(2)
                + x[0] * x[1]
                + (-4.0 + 4.0 * x[1].powi(2)) * x[1].powi(2),
        )
    }

    fn variable_bounds(&self) -> Array2<f64> {
        array![[-3.0, 3.0], [-2.0, 2.0]]
    }

    // Note: We intentionally do NOT implement gradient() or hessian() 
    // to demonstrate pure derivative-free optimization with COBYLA.
    // This is realistic for many real-world problems where derivatives
    // are unavailable, unreliable, or expensive to compute.
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Six-Hump Camel Function Optimization with COBYLA");
    println!("=================================================");
    println!("Using derivative-free optimization - no gradients computed!");
    println!();

    let problem: SixHumpCamel = SixHumpCamel;
    
    // Configure COBYLA with custom parameters for demonstration
    let cobyla_config = COBYLABuilder::default()
        .max_iter(500)                     // Increase iterations for better convergence
        .initial_step_size(0.3)            // Larger initial step for exploration
        .ftol_rel(1e-8)                    // Tight relative tolerance  
        .ftol_abs(1e-10)                   // Tight absolute tolerance
        .build();
    
    let params: OQNLPParams = OQNLPParams {
        iterations: 200,                             // Global search iterations
        wait_cycle: 12,                              // Wait cycles before expanding
        threshold_factor: 0.15,                      // Merit filter threshold
        distance_factor: 0.8,                        // Distance filter factor
        population_size: 400,                        // Scatter search population
        local_solver_type: LocalSolverType::COBYLA,  // Use COBYLA solver
        local_solver_config: cobyla_config,          // COBYLA configuration
        seed: 0,                                     // Random seed for reproducibility
    };

    println!("COBYLA Configuration:");
    println!("- Maximum iterations: 500");
    println!("- Initial step size: 0.3");
    println!("- Relative tolerance: 1e-8");
    println!("- Absolute tolerance: 1e-10");
    println!();
    
    println!("Global Search Parameters:");
    println!("- Population size: {}", params.population_size);
    println!("- Global iterations: {}", params.iterations);
    println!("- Using derivative-free COBYLA local solver");
    println!();

    let mut oqnlp: OQNLP<SixHumpCamel> = OQNLP::new(problem, params)?.verbose();
    
    println!("Starting optimization...");
    let solution_set: SolutionSet = oqnlp.run()?;

    println!("\n{}", solution_set);
    
    // Analysis of results
    if let Some(best) = solution_set.best_solution() {
        println!("Analysis:");
        println!("========");
        println!("Best objective found: {:.8}", best.objective);
        println!("Known global minimum: -1.0316");
        println!("Error from global minimum: {:.6}", (best.objective + 1.0316).abs());
        
        let point = &best.point;
        println!("Solution point: [{:.6}, {:.6}]", point[0], point[1]);
        println!("Known global minima:");
        println!("  (±0.0898, ∓0.7126) = -1.0316");
        
        let distance_to_min1 = ((point[0] - 0.0898).powi(2) + (point[1] + 0.7126).powi(2)).sqrt();
        let distance_to_min2 = ((point[0] + 0.0898).powi(2) + (point[1] - 0.7126).powi(2)).sqrt();
        let min_distance = distance_to_min1.min(distance_to_min2);
        
        println!("Distance to nearest known minimum: {:.6}", min_distance);
        
        if min_distance < 0.01 {
            println!("✓ Successfully found global minimum!");
        } else if best.objective < -1.0 {
            println!("✓ Found good solution near global minimum");
        } else {
            println!("? May have found local minimum");
        }
    }
    
    println!();
    println!("This example demonstrates:");
    println!("- Derivative-free optimization using COBYLA");
    println!("- No gradient or Hessian computation required");
    println!("- Suitable for noisy or non-differentiable functions");
    println!("- COBYLA's ability to handle complex optimization landscapes");
    
    Ok(())
}