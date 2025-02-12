use std::env;

fn main() {
    if env::var("CARGO_FEATURE_RAYON").is_ok() {
        println!(
            "cargo:warning=[Performance Notice] The 'rayon' feature is enabled. In debug builds or low workload scenarios, the overhead of thread management might reduce performance. For production builds, consider benchmarking with and without 'rayon'."
        );
    }
}
