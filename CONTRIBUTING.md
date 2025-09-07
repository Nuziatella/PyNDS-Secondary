# Contributing to PyNDS

Welcome to the PyNDS contributor's guide! You're about to become part of a time machine that brings Nintendo DS emulation to Python. Whether you're fixing bugs, adding features, or just making the code more awesome, we're excited to have you on board!

## What Makes PyNDS Special?

PyNDS isn't just another emulator wrapper - it's a gateway to nostalgia, reinforcement learning adventures, and automated gaming magic. We're building something that lets developers control Nintendo DS and Game Boy Advance games programmatically, and we need your help to make it even better!

**Note**: This is a secondary development repository that enhances the original [PyNDS project](https://github.com/unexploredtest/PyNDS) with improved documentation, testing, and development workflow. Contributions here will be considered for upstream integration into the main reposity, [PyNDS project](https://github.com/unexploredtest/PyNDS).

## Getting Started

### Prerequisites

Before you start contributing, make sure you have:

- **Python 3.11+** (because we're not living in the stone age)
- **Git** (for version control)
- **A sense of adventure and accepting we all make mistakes** (optional but highly recommended)

### Setting Up Your Development Environment

1. **Fork and Clone** the repository:
   ```bash
   git clone https://github.com/your-username/PyNDS.git
   cd PyNDS
   ```

2. **Create a Virtual Environment** (because isolation is key):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies** (the digital toolbox):
   ```bash
   pip install -e .
   pip install -r tests/requirements.txt
   ```

   Or install from the original repository:
   ```bash
   pip install pynds
   ```

4. **Install Pre-commit Hooks** (for automatic code quality):
   ```bash
   pre-commit install
   ```

## How to Contribute

### Reporting Bugs

Found a bug? Don't panic! Here's how to report it:

1. **Check if it's already reported!** - Search existing issues first
2. **Create a new issue with the following format**:
   - Clear, descriptive title
   - Steps to reproduce the bug
   - Expected vs actual behavior
   - System information (OS, Python version, etc.)
   - Screenshots or error messages if relevant

### Suggesting Features

Have an idea for a new feature?

1. **Check existing feature requests first**
2. **Create a new issue  with**:
   - Clear description of the feature
   - Use cases and benefits
   - Any implementation ideas you have
   - Examples of how it would be used

### Code Contributions

Ready to write some code? Here's the process:

1. **Create a new branch** from `main`:
   ```bash
   git checkout -b feature/your-awesome-feature
   # or
   git checkout -b fix/annoying-bug
   ```

2. **Make your changes** following the creater, unexploredtest, coding standards
3. **Write tests** for your changes (because code needs quality assurances)
4. **Run the test suite** to make sure everything works
5. **Submit a pull request** with a clear description

## Coding Standards

### Code Style

We follow Python's PEP 8 style guide with some PyNDS-specific tweaks:

- **Line length**: 88 characters (Black's default)
- **Indentation**: 4 spaces (no tabs, we're not coding savages)
- **Imports**: Use `isort` for consistent import ordering
- **Formatting**: Use `black` for automatic code formatting
- **Linting**: Use `ruff` for fast, modern linting

### Docstrings

All functions, classes, and modules must have docstrings in **NumPy style**:

```python
def awesome_function(param1: str, param2: int = 42) -> bool:
    """Do something awesome with the given parameters.

    This function performs digital magic that would make even
    the most skeptical developer believe in the power of Python.

    Parameters
    ----------
    param1 : str
        The first parameter (description of what it does)
    param2 : int, optional
        The second parameter (description), by default 42

    Returns
    -------
    bool
        True if awesome, False if not awesome

    Raises
    ------
    ValueError
        If param1 is empty or param2 is negative

    Examples
    --------
    >>> result = awesome_function("hello", 10)
    >>> print(result)
    True
    """
    # Your awesome code here
    pass
```

### Type Hints

Use type hints everywhere! They make the code more readable and help catch bugs! Seriously one of my biggest pet peeves is when things do not provide clear type hints! I CAN'T READ MINDS:

```python
from typing import Union, List, Optional

def process_rom_data(data: bytes, is_gba: bool = False) -> Union[List[int], None]:
    """Process ROM data with proper type safety."""
    # Your code here
    pass
```

## Testing

### Running Tests

We have a comprehensive test suite that you should run before submitting changes:

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=pynds --cov-report=html

# Run specific test file
python -m pytest tests/test_pynds.py

# Run tests with specific markers
python -m pytest -m "unit"
python -m pytest -m "integration"
```

### Writing Tests

When adding new features, write tests that cover:

- **Happy path** - Normal usage scenarios
- **Edge cases** - Boundary conditions and unusual inputs
- **Error conditions** - What happens when things go wrong
- **Integration** - How your code works with other components

### Test Structure

Follow our test organization:

```
tests/
├── conftest.py          # Shared fixtures and configuration
├── test_pynds.py        # Core PyNDS functionality
├── test_button.py       # Button input interface
├── test_memory.py       # Memory access interface
├── test_window.py       # Window display interface
└── requirements.txt     # Test-specific dependencies
```

## Development Tools

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### Code Formatting

```bash
# Format code with Black
black pynds/ tests/

# Sort imports with isort
isort pynds/ tests/

# Lint with ruff
ruff check pynds/ tests/
```

### Coverage Requirements

- **Minimum coverage**: 80%
- **Target coverage**: 90%+
- **Critical paths**: Must have 100% coverage

## Pull Request Process

### Before Submitting

1. **Run the full test suite**:
   ```bash
   python -m pytest --cov=pynds --cov-fail-under=80
   ```

2. **Check code formatting**:
   ```bash
   black --check pynds/ tests/
   isort --check-only pynds/ tests/
   ruff check pynds/ tests/
   ```

3. **Update documentation** if needed
4. **Add tests** for new functionality
5. **Update CHANGELOG.md** if applicable

### Pull Request Template

When creating a PR, include:

- **Description**: What changes you made and why
- **Type**: Bug fix, feature, documentation, etc.
- **Testing**: How you tested your changes
- **Breaking changes**: Any API changes that might affect users
- **Screenshots**: If applicable (especially for UI changes)

### Review Process

1. **Automated checks** must pass (tests, linting, coverage)
2. **Code review** by maintainers
3. **Discussion** of any requested changes
4. **Approval** and merge

## PyNDS-Specific Guidelines

### ROM Files

- **Never commit ROM files** to the repository
- **Use test ROMs** for development and testing
- **Respect copyright** - only use legally obtained ROMs
- **Document ROM requirements** in test files

### Emulator Integration

- **Mock external dependencies** in tests when possible
- **Handle initialization failures** gracefully
- **Provide clear error messages** for common issues
- **Test both NDS and GBA modes** when applicable

### Performance Considerations

- **Profile memory usage** for large operations
- **Consider frame rate** for real-time applications
- **Optimize hot paths** in the emulation loop
- **Document performance characteristics**

## Common Issues and Solutions

### Import Errors

```bash
# If you get import errors, make sure you're in the right environment
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -e .
```

### Test Failures

```bash
# If tests fail, check the specific error
python -m pytest tests/test_specific.py -v

# Run with more verbose output
python -m pytest tests/ -v -s
```

### Code Style Issues

```bash
# Let Black fix formatting issues
black pynds/ tests/

# Let isort fix import ordering
isort pynds/ tests/

# Let ruff fix linting issues
ruff check pynds/ tests/ --fix
```

## Community Guidelines

### Code of Conduct

- **Be respectful** and inclusive
- **Be constructive** in feedback
- **Be patient** with newcomers
- **Be helpful** and supportive

### Communication

- **Use clear, descriptive commit messages**
- **Write helpful PR descriptions**
- **Respond to feedback** promptly
- **Ask questions** when you need help

### Recognition

Contributors will be recognized in:
- **CONTRIBUTORS.md** file
- **Release notes** for significant contributions
- **GitHub contributors** page

## Additional Resources

- **PyNDS Repository**: [https://github.com/unexploredtest/PyNDS](https://github.com/unexploredtest/PyNDS)
- **NooDS Repository**: [https://github.com/Hydr8gon/NooDS](https://github.com/Hydr8gon/NooDS)
- **Python Style Guide**: [PEP 8](https://peps.python.org/pep-0008/)
- **NumPy Docstring Guide**: [https://numpydoc.readthedocs.io/](https://numpydoc.readthedocs.io/)

## Getting Help

If you need help:

1. **Check the documentation** first
2. **Search existing issues** for similar problems
3. **Ask in discussions** for general questions
4. **Create an issue** for bugs or feature requests
5. **Join our community** (if we have one)

## Ready to Contribute?

Great! Here's your first contribution checklist:

- [ ] Fork the repository
- [ ] Set up your development environment
- [ ] Read through the codebase
- [ ] Find an issue to work on (or create one)
- [ ] Create a branch and make your changes
- [ ] Write tests for your changes
- [ ] Run the test suite
- [ ] Submit a pull request

**Welcome to the PyNDS community! Let's build something amazing together!**

---

*"In the digital realm, every contribution matters - even the smallest bug fix can make someone's day!"* - The PyNDS Philosophy
