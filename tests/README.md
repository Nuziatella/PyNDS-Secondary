# PyNDS Test Suite

Welcome to the PyNDS test suite - your quality assurance laboratory! This directory contains all the tests to ensure that PyNDS works correctly and doesn't break.

## Test Structure

- `test_pynds.py` - Tests for the main PyNDS class (core emulation functionality)
- `test_button.py` - Tests for button input and touch screen functionality
- `test_memory.py` - Tests for memory read/write operations
- `test_window.py` - Tests for pygame-based window display
- `conftest.py` - Shared fixtures and test configuration
- `requirements.txt` - Testing dependencies

## Running Tests

### Prerequisites
Install the testing dependencies:
```bash
pip install -r tests/requirements.txt
```

### Basic Test Run
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_pynds.py

# Run specific test
pytest tests/test_pynds.py::TestPyNDSInitialization::test_init_with_valid_nds_file
```

### Test Categories
```bash
# Run only unit tests (fast)
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run only tests that don't require ROMs
pytest -m "not requires_rom"
```

## Test Philosophy

Our tests follow these principles:

1. **Mock External Dependencies** - We mock the C++ bindings to test Python logic without requiring actual ROMs
2. **Comprehensive Coverage** - Test both happy paths and error conditions
3. **Fast Execution** - Tests should run quickly for rapid development feedback
4. **Clear Documentation** - Each test explains what it's testing and why
5. **Maintainable** - Tests are easy to understand and modify

## Writing New Tests

When adding new tests:

1. **Follow the naming convention**: `test_*.py` files, `test_*` functions
2. **Use descriptive names**: `test_press_key_valid` not `test_press_key_1`
3. **Mock external dependencies**: Don't require real ROMs or C++ libraries
4. **Test edge cases**: Invalid inputs, error conditions, boundary values
5. **Add docstrings**: Explain what the test is verifying
6. **Use appropriate markers**: Mark slow tests, integration tests, etc.

## Mock Strategy

Since PyNDS depends on C++ bindings, we use extensive mocking:

- **Mock the `cnds` module** - Simulates the C++ emulator
- **Mock pygame** - Simulates the display system
- **Mock file operations** - Simulates ROM file access
- **Use fixtures** - Provide consistent test data

This allows us to test the Python logic without requiring:
- Actual ROM files
- Compiled C++ libraries
- Display hardware
- Complex setup

## Continuous Integration

The test suite is designed to run in CI environments:

- **No external dependencies** - All tests use mocks
- **Fast execution** - Complete suite runs in under a minute
- **Clear output** - Verbose reporting for debugging failures
- **Timeout protection** - Tests won't hang indefinitely

## Contributing Tests

When contributing to PyNDS:

1. **Add tests for new features** - Every new function should have tests
2. **Add tests for bug fixes** - Prevent regressions
3. **Update existing tests** - When changing behavior
4. **Run the full suite** - Before submitting changes
5. **Keep tests passing** - Don't commit failing tests

Remember: Good tests are like good documentation - they explain how the code should work and catch bugs before they reach users!

---
