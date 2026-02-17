# Contributing to Clinical Safeguard

Thank you for your interest in contributing to Clinical Safeguard! This document provides guidelines for contributing to
the project.

## Table of Contents

- [Development Setup](#development-setup)
- [Adding a New Pipeline Stage](#adding-a-new-pipeline-stage)
- [Extending Keyword Resources](#extending-keyword-resources)
- [Testing Requirements](#testing-requirements)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)

## Development Setup

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Git

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/JMasr/clinical-safeguard.git
cd clinical-safeguard

# Create virtual environment
make create_environment
source .venv/bin/activate

# Install dependencies (including dev dependencies)
make requirements-dev

# Create .env file for tests
cat > .env << EOF
HF_TOKEN=hf_your_token_here
SAFEGUARD_TIMEOUT=10
EOF
```

### Verify Setup

```bash
# Run all tests
make test

# Check linting
make lint
```

## Adding a New Pipeline Stage

The pipeline is designed for extensibility. Follow these steps to add a new stage:

### Step 1: Create the Stage Class

Create a new file in `src/stages/` that inherits from `GuardrailStage`:

```python
# src/stages/my_new_stage.py

from __future__ import annotations

import logging
from src.core.base import GuardrailStage
from src.core.exceptions import StageExecutionError
from src.models import Label, PromptInput, StageResult

logger = logging.getLogger(__name__)


class MyNewStage(GuardrailStage):
    """
    Description of what this stage detects.
    
    Document:
    - What labels it can return
    - When it short-circuits
    - Any external dependencies
    """

    _NAME = "my_new_stage"

    def __init__(
            self,
            # Add constructor parameters here
            threshold: float = 0.75,
    ) -> None:
        self._threshold = threshold
        logger.info("MyNewStage configured — threshold=%.2f", threshold)

    @property
    def name(self) -> str:
        return self._NAME

    def process(self, prompt: PromptInput) -> StageResult:
        """
        Contract:
        - MUST return StageResult (never None)
        - MUST NOT raise exceptions (wrap in StageExecutionError)
        - MUST NOT mutate the prompt
        """
        try:
            return self._evaluate(prompt)
        except StageExecutionError:
            raise
        except Exception as exc:
            raise StageExecutionError(self._NAME, exc) from exc

    def _evaluate(self, prompt: PromptInput) -> StageResult:
        # Your detection logic here

        # Example: return Valid if no issues detected
        return StageResult(
            stage_name=self._NAME,
            label=Label.VALID,
            confidence=1.0,
            triggered_by=None,
            short_circuit=False,
        )
```

### Step 2: Register the Stage

Add your stage to the `STAGE_REGISTRY` in `src/stages/__init__.py`:

```python
# src/stages/__init__.py

from __future__ import annotations

from .attack_detection import AttackDetectionStage
from .deterministic import DeterministicStage
from .semantic import SemanticBERTStage
from .my_new_stage import MyNewStage  # Add import

STAGE_REGISTRY: frozenset[str] = frozenset(
    {
        "src.stages.deterministic.DeterministicStage",
        "src.stages.semantic.SemanticBERTStage",
        "src.stages.attack_detection.AttackDetectionStage",
        "src.stages.my_new_stage.MyNewStage",  # Add to registry
    }
)
```

> **Important:** The registry validation happens at startup. Any `_target_` in Hydra configs that isn't in this set will
> cause a `ResourceLoadError` — preventing arbitrary code execution via malformed configs.

### Step 3: Create a Pipeline Configuration

Add a new profile in `conf/pipeline/`:

```yaml
# conf/pipeline/with_my_stage.yaml

stages:
  - _target_: src.stages.deterministic.DeterministicStage
    keywords_crisis_path: ${paths.crisis}
    keywords_malign_path: ${paths.malign}
    bypass_patterns_path: ${paths.bypass}

  - _target_: src.stages.my_new_stage.MyNewStage
    threshold: 0.80
```

### Step 4: Write Tests

Create comprehensive tests in `tests/test_my_new_stage.py`:

```python
# tests/test_my_new_stage.py

from __future__ import annotations

import pytest

from src.models import Label, PromptInput
from src.stages.my_new_stage import MyNewStage


class TestMyNewStageBasics:
    def test_stage_name(self) -> None:
        stage = MyNewStage()
        assert stage.name == "my_new_stage"

    def test_valid_input_returns_valid(self) -> None:
        stage = MyNewStage()
        prompt = PromptInput(text="Normal input text")

        result = stage.process(prompt)

        assert result.label == Label.VALID
        assert result.stage_name == "my_new_stage"


class TestMyNewStageDetection:
    # Add detection-specific tests
    pass


class TestMyNewStageErrorHandling:
    def test_exception_wrapped_in_stage_execution_error(self) -> None:
        # Test that internal errors are properly wrapped
        pass
```

### Stage Design Guidelines

1. **Stateless Processing**: Stages may hold pre-loaded resources but must not mutate `PromptInput`
2. **Never Raise**: Always wrap exceptions in `StageExecutionError`
3. **Short-Circuit Rules**:
    - `Crisis` and `Malign` labels MUST set `short_circuit=True`
    - `Valid` MUST set `short_circuit=False`
4. **Lazy Loading**: For ML models, load on first use, not in `__init__`
5. **Thread Safety**: Use locks for shared resources under concurrent access

## Extending Keyword Resources

### Adding Crisis Keywords

Edit `resources/keywords_crisis.yaml`:

```yaml
categories:
  # Each category maps to a C-SSRS level
  new_category_name:
    cssrs_level: 1-10  # Severity level
    keywords:
      - keyword one
      - keyword two
      - multi word phrase
```

**Requirements:**

- Cite clinical sources for new keywords
- Map to appropriate C-SSRS level (1-10)
- Do NOT remove entries without clinical review

### Adding Malign Keywords

Edit `resources/keywords_malign.yaml`:

```yaml
categories:
  harm_to_others:
    keywords:
      - new harmful phrase

  policy_violation:
    keywords:
      - policy violating term
```

### Adding Bypass Patterns

Edit `resources/bypass_patterns.yaml`:

```yaml
patterns:
  - id: BYPASS_NEW_PATTERN
    pattern: "regex pattern here"
    description: "What this pattern detects"
    references:
      - "Citation or source URL"
```

**Pattern Guidelines:**

- Use stable IDs (format: `BYPASS_XXX`)
- Patterns are case-insensitive
- Test against false positives
- Document the attack vector being addressed

## Testing Requirements

### Minimum Requirements

- All new code must have tests
- Target: 100% line coverage
- All tests must pass before merging

### Running Tests

```bash
# Full test suite
make test

# Specific file
python -m pytest tests/test_my_new_stage.py -v

# With coverage report
python -m pytest tests/ --cov=src --cov-report=html
```

### Test Categories

1. **Unit Tests**: Test individual functions/methods in isolation
2. **Integration Tests**: Test stage interactions with the pipeline
3. **Contract Tests**: Verify Pydantic models and API contracts
4. **HTTP Tests**: Test FastAPI endpoints with TestClient

### Test Fixtures

Use the fixtures in `tests/conftest.py`:

```python
# Available fixtures
@pytest.fixture
def valid_prompt() -> PromptInput:
    """Safe, benign prompt for testing Valid paths"""


@pytest.fixture
def crisis_prompt() -> PromptInput:
    """Prompt containing crisis keywords"""


@pytest.fixture
def malign_prompt() -> PromptInput:
    """Prompt containing malicious content"""


# Helper function
def make_stage_result(
        label: Label,
        stage_name: str = "test_stage",
        confidence: float = 0.99,
        triggered_by: str | None = None,
        short_circuit: bool = True,
) -> StageResult:
    """Factory for StageResult instances in tests"""
```

### Testing ML Stages

For stages that use ML models, inject a mock factory:

```python
from unittest.mock import MagicMock


def test_my_ml_stage() -> None:
    # Create mock classifier
    mock_classifier = MagicMock(
        return_value=[{"label": "SAFE", "score": 0.95}]
    )
    mock_factory = MagicMock(return_value=mock_classifier)

    # Inject mock
    stage = MyMLStage(
        model_id="test/model",
        pipeline_factory=mock_factory,
    )

    result = stage.process(PromptInput(text="test"))

    assert result.label == Label.VALID
    mock_classifier.assert_called_once()
```

## Code Style

### Formatting

We use `ruff` for linting and formatting:

```bash
# Check style
make lint

# Auto-fix issues
make format
```

### Guidelines

1. **Type Hints**: All public functions must have type annotations
2. **Docstrings**: Use Google-style docstrings for public APIs
3. **Imports**:
    - Use `from __future__ import annotations`
    - Group: stdlib, third-party, local
4. **Naming**:
    - Classes: `PascalCase`
    - Functions/variables: `snake_case`
    - Constants: `UPPER_SNAKE_CASE`
    - Private: `_leading_underscore`

### Example

```python
from __future__ import annotations

import logging
from typing import Sequence

from pydantic import BaseModel

from src.core.base import GuardrailStage

logger = logging.getLogger(__name__)


class MyClass:
    """
    Brief description.

    Longer description if needed.

    Attributes:
        name: Description of the attribute.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def process(self, items: Sequence[str]) -> list[str]:
        """
        Process a sequence of items.

        Args:
            items: The items to process.

        Returns:
            Processed items as a list.

        Raises:
            ValueError: If items is empty.
        """
        if not items:
            raise ValueError("items cannot be empty")
        return list(items)
```

## Pull Request Process

### Before Submitting

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes** following the guidelines above

3. **Run quality checks**:
   ```bash
   make lint
   make test
   ```

4. **Commit with clear messages**:
   ```bash
   git commit -m "feat(stages): add MyNewStage for X detection"
   ```

### Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

Types:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Adding tests
- `refactor`: Code refactoring
- `chore`: Maintenance tasks

### PR Checklist

- [ ] Tests pass locally (`make test`)
- [ ] Linting passes (`make lint`)
- [ ] New code has tests
- [ ] Documentation updated if needed
- [ ] No sensitive data in commits
- [ ] Commit messages follow convention

### Review Process

1. Submit PR against `main` branch
2. CI will run tests and linting
3. Maintainer will review code
4. Address feedback if any
5. Maintainer merges after approval

## Questions?

If you have questions about contributing:

1. Check existing issues and PRs
2. Open a discussion if needed
3. Tag maintainers for urgent matters

Thank you for contributing to making LLM applications safer! 🛡️

