# Contributing to TypeScript SLM

Thank you for your interest in contributing to TypeScript SLM! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Code Style](#code-style)
- [Documentation](#documentation)

## Code of Conduct

This project follows a professional code of conduct:

- **Be respectful**: Treat all contributors with respect
- **Be constructive**: Provide helpful feedback and suggestions
- **Be collaborative**: Work together to improve the project
- **Be professional**: Maintain a productive and welcoming environment

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- GitHub account
- Basic understanding of machine learning and PyTorch

### Areas for Contribution

We welcome contributions in the following areas:

1. **Model Improvements**
   - New model architectures or variants
   - Training optimization techniques
   - Hyperparameter tuning

2. **Data Quality**
   - Improved filtering algorithms
   - New data sources
   - Dataset quality metrics

3. **Platform Support**
   - Additional hardware platforms
   - Platform-specific optimizations
   - Cloud provider integrations

4. **Documentation**
   - Tutorial improvements
   - API documentation
   - Example notebooks

5. **Bug Fixes**
   - Issue resolution
   - Performance improvements
   - Edge case handling

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/slm-typescript-model.git
cd slm-typescript-model

# Add upstream remote
git remote add upstream https://github.com/sylvester-francis/slm-typescript-model.git
```

### 2. Create Virtual Environment

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt # If available

# Install pre-commit hooks (if configured)
pre-commit install
```

### 4. Set Up Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your tokens
# GITHUB_TOKEN, HF_TOKEN, etc.
```

### 5. Verify Installation

```bash
# Run environment check
python scripts/check_environment.py

# Run tests (if available)
pytest
```

## Contributing Guidelines

### Creating Issues

Before creating an issue:

1. **Search existing issues** to avoid duplicates
2. **Use issue templates** when available
3. **Provide detailed information**:
   - Clear description of the problem or feature
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Environment details (OS, Python version, GPU)
   - Relevant logs or error messages

### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Make focused commits**:
   - One logical change per commit
   - Write clear commit messages
   - Reference issues in commits (e.g., "Fix #123")

3. **Follow code style** (see [Code Style](#code-style))

4. **Update documentation** for any user-facing changes

5. **Add tests** for new functionality

### Commit Message Format

Use clear, descriptive commit messages:

```
<type>: <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test additions or changes
- `chore`: Build process or auxiliary tool changes

**Example**:
```
feat: Add support for Llama 3 base model

Implement training pipeline for Llama 3 architecture:
- Add model loading with proper tokenizer handling
- Configure LoRA targets for Llama 3 architecture
- Update documentation with usage examples

Closes #42
```

## Pull Request Process

### Before Submitting

1. **Update your branch**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests**:
   ```bash
   python scripts/check_environment.py
   pytest # If tests are available
   ```

3. **Test on multiple platforms** (if applicable):
   - Mac M-series (MPS)
   - Google Colab (CUDA)
   - Linux GPU

4. **Update documentation**:
   - Update README.md if needed
   - Update relevant docs/ files
   - Add docstrings to new functions
   - Update CHANGELOG (if maintained)

### Submitting Pull Request

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create pull request** on GitHub

3. **Fill out PR template** with:
   - Description of changes
   - Related issues
   - Testing performed
   - Platform compatibility
   - Breaking changes (if any)

4. **Wait for review**:
   - Address reviewer feedback
   - Update PR as requested
   - Keep discussion focused and professional

### PR Review Criteria

Your PR will be reviewed for:

- **Functionality**: Does it work as intended?
- **Code quality**: Is it well-written and maintainable?
- **Testing**: Are changes adequately tested?
- **Documentation**: Is it properly documented?
- **Compatibility**: Works across supported platforms?
- **Performance**: No significant performance regressions?

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_training.py

# Run with coverage
pytest --cov=scripts tests/
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use descriptive test names
- Test edge cases and error conditions
- Mock external dependencies (GitHub API, HuggingFace, etc.)

### Manual Testing

For changes affecting training:

1. **Test with small dataset**:
   ```bash
   python cli.py train --data data/processed/train_small.jsonl --max-samples 100
   ```

2. **Verify on target platform**:
   - Test on Colab if changing CUDA code
   - Test on Mac if changing MPS code

3. **Check memory usage**:
   - Monitor GPU memory during training
   - Ensure no memory leaks

## Code Style

### Python Style

- Follow **PEP 8** style guide
- Use **4 spaces** for indentation
- Maximum line length: **100 characters**
- Use **type hints** for function signatures
- Write **docstrings** for all public functions

### Formatting Tools

```bash
# Format code with black
black scripts/ cli.py

# Sort imports with isort
isort scripts/ cli.py

# Check with flake8
flake8 scripts/ cli.py
```

### Docstring Format

Use Google-style docstrings:

```python
def train_model(model_name: str, data_path: str, epochs: int = 3) -> None:
    """Train a TypeScript SLM model.

    Args:
        model_name: HuggingFace model identifier
        data_path: Path to training data JSONL file
        epochs: Number of training epochs

    Returns:
        None

    Raises:
        FileNotFoundError: If data_path does not exist
        ValueError: If epochs < 1
    """
    pass
```

### Variable Naming

- Use **snake_case** for variables and functions
- Use **UPPER_CASE** for constants
- Use **PascalCase** for classes
- Use descriptive names (avoid single letters except in loops)

## Documentation

### README Updates

Update README.md when:
- Adding new features
- Changing CLI interface
- Modifying installation process
- Updating requirements

### API Documentation

- Document all public functions and classes
- Include usage examples
- Explain parameters and return values
- Document exceptions raised

### Tutorial Documentation

Place tutorials in `docs/`:
- Use clear, step-by-step instructions
- Include code examples
- Add screenshots when helpful
- Test all commands and code samples

### Changelog

Update CHANGELOG.md (if maintained) with:
- Version number
- Release date
- Added features
- Bug fixes
- Breaking changes

## Development Workflow

### Typical Workflow

1. Check existing issues or create new one
2. Fork repository and create feature branch
3. Make changes with tests and documentation
4. Run tests and verify on target platforms
5. Commit with clear messages
6. Push to your fork
7. Create pull request
8. Address review feedback
9. Merge after approval

### Getting Help

- **Questions**: Open a discussion on GitHub
- **Bugs**: Create an issue with details
- **Features**: Propose in an issue first
- **Urgent**: Tag maintainers in issue

## License

By contributing to TypeScript SLM, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors are recognized in:
- GitHub contributors list
- Release notes
- Project documentation (for significant contributions)

## Additional Resources

- [GitHub Flow Guide](https://guides.github.com/introduction/flow/)
- [Writing Good Commit Messages](https://chris.beams.io/posts/git-commit/)
- [Python Style Guide (PEP 8)](https://www.python.org/dev/peps/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

---

Thank you for contributing to TypeScript SLM!
