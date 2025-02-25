# CONTRIBUTING

Thank you for considering contributing to the dendritic_modeling package! We value your input and aim to make the contribution process as smooth as possible. Please follow the guidelines below to ensure effective collaboration. 

## Reporting Issues

Before starting any work, please open a new issue to discuss the changes you intend to make. This helps us track progress and ensures that efforts are aligned with project priorities.

## Branching Strategy

We adhere to the [Successful Git Branching](https://nvie.com/posts/a-successful-git-branching-model/) Model. Our primary branches are:
- `main`: Contains production-ready code.
- `develop`: Serves as the integration branch for features.

For new features or fixes, create a feature branch from the `develop` branch. Name your branch descriptively, such as `iss[issue number]_[short description]`. 

## Development Workflow

### Fork the Repository

If you are not part of the core developers, fork the repository to your GitHub account. This creates a copy of the repository under your username.

### Clone the Repository

```bash
git clone https://github.com/your-username/dendritic_modeling.git
```

### Create a Feature Branch

```bash
git checkout -b feature/your-feature-name develop
```

### Implement Changes and Commit

For documentation changes, manually build the documentation to ensure accuracy. For code changes, run tests to verify functionality.

## Pull Request Process


- Open a Pull Request (PR):
  - Base branch: develop
  - Compare branch: your-feature-branch

- Provide a detailed description of the changes and their purpose.
- Monitor CI/CD Pipelines:
  - Ensure all tests pass.
- Address any issues promptly.
- Engage in Code Review:
  - Respond to feedback from reviewers.
- Make necessary revisions.
- Merge:
  - Once approved, a maintainer will merge your PR into develop.

**Note**: For substantial changes, consult with core developers to confirm alignment with project priorities. Submitting a PR early, marked as [WIP] (Work in Progress), allows for continuous integration testing and early feedback. 