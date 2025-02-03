use std::env;

fn main() {
    if env::var("CARGO_FEATURE_RAYON").is_ok() {
        println!("cargo:warning=Using the 'rayon' feature may lead to non-reproducible results as parallel execution does not guarantee deterministic ordering.");
    }
}
