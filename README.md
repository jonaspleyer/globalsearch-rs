<p align="center">
    <img
        width="500"
        src="https://raw.githubusercontent.com/GermanHeim/globalsearch-rs/main/media/logo.png"
        alt="GlobalSearch-rs"
    />
    <p align="center">
        Global optimization with scatter search and local NLP solvers written in Rust
    </p>
</p>

`globalsearch-rs`: Rust implementation of the _OQNLP_ (_OptQuest/NLP_) algorithm from "Scatter Search and Local NLP Solvers: A Multistart Framework for Global Optimization" by Ugray et al. (2007). Combines scatter search metaheuristics with local minimization for global optimization of nonlinear problems.

Similar to MATLAB's `GlobalSearch` [2], using argmin, rayon and ndarray.

## Features

- 🎯 Multistart heuristic framework for global optimization

- 📦 Local optimization using the argmin crate [3]

- 🚀 Parallel execution of initial stage using Rayon

## Installation

1. Install Rust toolchain using [rustup](https://rustup.rs/).
2. Clone repository:

   ```bash
    git clone https://github.com/GermanHeim/globalsearch-rs.git
    cd globalsearch-rs
   ```

3. Build the project:

   ```bash
    cargo build --release
   ```

## Usage

1. Define a problem by implementing the `Problem` trait.

   ```rust
    use ndarray::{array, Array1, Array2};
    use globalsearch_rs::problem::Problem;

    pub struct MinimizeProblem;
    impl Problem for MinimizeProblem {
        fn objective(&self, x: &Array1<f64>) -> Result<f64> {
            Ok(
                ..., // Your objective function here
            )
        }

        fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>> {
            Ok(array![
                ..., // Gradient of your objective function here
            ])
        }

        fn variable_bounds(&self) -> Array2<f64> {
            array![[..., ...]. [..., ...]] // Lower and upper bounds for each variable
        }
   }
   ```

   Where the `Problem` trait is defined as:

   ```rust
    pub trait Problem {
        fn objective(&self, x: &Array1<f64>) -> Result<f64>;
        fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>>;
        fn variable_bounds(&self) -> Array2<f64>;
    }
   ```

   > 🔴 **Note:** Variable bounds are only used in the scatter search phase of the algorithm. The local solver is unconstrained (See [argmin issue #137](https://github.com/argmin-rs/argmin/issues/137)) and therefor can return solutions out of bounds.

2. Set OQNLP parameters

   ```rust
    use globalsearch_rs::types::{LocalSolverType, OQNLPParams, SteepestDescentBuilder};

    let params: OQNLPParams = OQNLPParams {
        total_iterations: 1000,
        stage_1_iterations: 200,
        wait_cycle: 20,
        threshold_factor: 0.2,
        distance_factor: 0.75,
        population_size: 10,
        local_solver_type: LocalSolverType::SteepestDescent,
        local_solver_config: SteepestDescentBuilder::default().build(),
    };
   ```

   Where `OQNLPParams` is defined as:

   ```rust
    pub struct OQNLPParams {
        pub total_iterations: usize,
        pub stage_1_iterations: usize,
        pub wait_cycle: usize,
        pub threshold_factor: f64,
        pub distance_factor: f64,
        pub population_size: usize,
        pub local_solver_type: LocalSolverType,
        pub local_solver_config: LocalSolverConfig,
    }
   ```

   And `LocalSolverType` is defined as:

   ```rust
    pub enum LocalSolverType {
        LBFGS,
        NelderMead,
        SteepestDescent,
    }
   ```

   You can also modify the local solver configuration for each type of local solver. See `types.rs` for more details.

3. Run the optimizer

   ```rust
   use oqnlp::{OQNLP, OQNLPParams};

   fn main() -> anyhow::Result<()> {
        let problem = MinimizeProblem;
        let params: OQNLPParams = OQNLPParams {
                total_iterations: 1000,
                stage_1_iterations: 200,
                wait_cycle: 20,
                threshold_factor: 0.2,
                distance_factor: 0.75,
                population_size: 10,
                local_solver_type: LocalSolverType::SteepestDescent,
                local_solver_config: SteepestDescentBuilder::default().build(),
            };

        let mut optimizer = OQNLP::new(problem, params)?;
        let solution = optimizer.run()?;

        println!("Best solution:");
        println!("Point: {:?}", solution.point);
        println!("Objective: {}", solution.objective);
        Ok(())
   }
   ```

## Project Structure

```plaintext
src/
├── lib.rs # Module declarations
├── oqnlp.rs # Core OQNLP algorithm implementation
├── scatter_search.rs # Scatter search component
├── filters.rs # Merit and distance filtering logic
├── local_solver.rs # argmin-based local optimization
├── problem.rs # Problem trait
├── types.rs # Data structures and parameters
```

## Dependencies

- [argmin](https://github.com/argmin-rs/argmin)
- [ndarray](https://github.com/rust-ndarray/ndarray)
- [rayon](https://github.com/rayon-rs/rayon)
- [rand](https://github.com/rust-random/rand)
- [anyhow](https://github.com/dtolnay/anyhow)

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See [`LICENSE.txt`](https://github.com/GermanHeim/globalsearch-rs/blob/main/LICENSE.txt) for more information.

## References

[1] Zsolt Ugray, Leon Lasdon, John Plummer, Fred Glover, James Kelly, Rafael Martí, (2007) Scatter Search and Local NLP Solvers: A Multistart Framework for Global Optimization. INFORMS Journal on Computing 19(3):328-340. <http://dx.doi.org/10.1287/ijoc.1060.0175>

[2] GlobalSearch. The MathWorks, Inc. Available at: <https://www.mathworks.com/help/gads/globalsearch.html> (Accessed: 27 January 2025)

[3] Kroboth, S. argmin{}. Available at: <https://argmin-rs.org/> (Accessed: 25 January 2025)
