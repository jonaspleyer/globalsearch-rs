# Contributing to `globalsearch-rs`

Thank you for your interest in contributing to `globalsearch-rs`! Your help is greatly appreciated in making this project better. Below are the guidelines for contributing.

## 🛠️ How to Contribute

1. Report Bugs & Request Features

   If you find a bug or have a feature request, please open an issue and use the appropriate labels (e.g., "bug", "enhancement").

   Provide a clear and concise description of the problem or feature.

   Include relevant error messages, logs, and steps to reproduce (if applicable).

2. Fork & Clone the Repository

   Fork the repository on GitHub and clone it to your local machine:

   ```bash
   git clone https://github.com/[your-username]/globalsearch-rs.git
   cd globalsearch-rs
   ```

3. Create a Feature Branch

   Follow the convention: feature/short-description or bugfix/short-description

   ```bash
   git checkout -b feature/new-feature
   ```

4. Make Your Changes

   Implement your feature or bugfix following best practices.

   Ensure the code is well-structured and follows Rust’s idiomatic conventions.

   If applicable, update documentation and comments.

5. Run Tests & Format Code

   Ensure your code passes all tests:

   ```bash
   cargo test
   ```

   Format your code before committing:

   ```bash
    cargo fmt
    cargo clippy
   ```

6. Commit and Push

   Write clear and concise commit messages:

   ```bash
   git commit -m "Add feature: brief description"
   git push origin feature/new-feature
   ```

7. Create a Pull Request

   Navigate to your GitHub fork and click "New Pull Request".

   Provide a clear description of your changes and reference related issues if applicable.

   Request a review from maintainers.

## 🔍 Code Style & Guidelines

### General

- Follow Rust's best practices and idioms.

- Use `cargo fmt` for formatting (default `cargofmt.toml`).

- Run `cargo clippy` to catch warnings and inefficiencies.

### Documentation

Write clear, concise comments where necessary.

Add `cargo doc` comments to public functions and structs.

If a new feature is added, update the relevant documentation.

### Testing

Add tests for new functionality whenever possible.

Run `cargo test` to verify the tests. Ensure that your changes do not reduce the number of passing tests.

## 🤝 Community & Support

Join discussions on issues and pull requests.

If you need help, feel free to ask by opening a new issue.

## 📜 License

By contributing, you agree that your contributions will be licensed under the same MIT License as the project.

Thank you for contributing!
