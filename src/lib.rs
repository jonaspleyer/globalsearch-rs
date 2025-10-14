#![cfg_attr(not(doctest), doc = include_str!("../README.md"))]
#![doc(
    html_favicon_url = "https://raw.githubusercontent.com/GermanHeim/globalsearch-rs/main/media/favicon.png"
)]
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/GermanHeim/globalsearch-rs/main/media/favicon.png"
)]
pub mod filters;
pub mod local_solver;
// pub mod oqnlp;
pub mod problem;
// pub mod scatter_search;
pub mod types;

// #[cfg(feature = "checkpointing")]
// pub mod checkpoint;
