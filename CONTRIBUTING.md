# Contributing to SSD Vector Engine

Thank you for your interest in contributing to the SSD Vector Engine project!

## Spec-Driven Development

This project follows a **spec-driven development** approach. Before writing code:

1. **Create a specification** in the appropriate `specs/` directory
2. **Get the spec reviewed** by creating a PR
3. **Implement according to the approved spec**
4. **Link your implementation PR** to the spec

See [specs/README.md](specs/README.md) for detailed guidance.

## Development Process

### 1. For New Features

```bash
# 1. Create a feature spec
touch specs/features/your-feature.md
# Use the template from .github/spec-kit.yml

# 2. Create a PR for the spec
git checkout -b spec/your-feature
git add specs/features/your-feature.md
git commit -m "spec: Add specification for your-feature"
git push origin spec/your-feature

# 3. After spec is approved, implement
git checkout -b feat/your-feature
# ... implement your feature ...
git commit -m "feat: Implement your-feature (refs spec/your-feature)"
git push origin feat/your-feature
```

### 2. For New Components

```bash
# 1. Create a component spec
touch specs/components/your-component.md

# 2. Follow the same PR process as features
```

### 3. For Architecture Decisions

```bash
# 1. Create an ADR (Architecture Decision Record)
# Use sequential numbering: adr-001, adr-002, etc.
touch specs/architecture/adr-NNN-your-decision.md

# 2. Propose the ADR via PR
# 3. Get team consensus before marking as Accepted
```

## Spec Quality Guidelines

A good spec includes:

- **Clear purpose**: What problem does this solve?
- **Concrete examples**: Show, don't just tell
- **API/Interface definitions**: Be specific about contracts
- **Performance requirements**: Define measurable targets
- **Error handling**: What can go wrong and how to handle it
- **Testing strategy**: How will this be validated?
- **Dependencies**: What does this rely on?

## Code Quality Guidelines

- Follow language-specific style guides (C++, Python)
- Write tests for all new functionality
- Ensure performance meets spec requirements
- Update documentation when changing behavior
- Keep commits focused and well-described

## Pull Request Process

1. **Spec PRs**:
   - Must include all required sections
   - Need at least 1 approval
   - Should be reviewed for completeness and clarity

2. **Implementation PRs**:
   - Must reference the approved spec
   - Must include tests
   - Must pass CI/CD checks
   - Need at least 1 approval from a maintainer

## Questions?

- Check [specs/README.md](specs/README.md) for spec-writing guidance
- Review existing specs for examples
- Open an issue for questions or discussions

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see [LICENSE](LICENSE)).
