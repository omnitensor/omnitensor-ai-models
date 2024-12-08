# OmniTensor AI Models

This directory contains AI models, training scripts and evaluation tools for the OmniTensor ecosystem. 
These models are essential for enabling decentralized AI capabilities.

## Features

- **Pre-trained Models**: Includes state-of-the-art language, vision, and speech models.
- **Training Scripts**: Easy-to-use scripts for fine-tuning or training new models.
- **Inference Tools**: Simple APIs for deploying models in production.
- **Evaluation Metrics**: Evaluate model performance with built-in metrics.

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

For development dependencies:

```bash
pip install -r requirements-dev.txt
```

## Usage

Example usage of an AI model:

```python
from omnitensor_ai_models.models.language_model import LanguageModel

model = LanguageModel()
output = model.generate_text("What is OmniTensor?")
print(output)
```

## Development

To contribute, please ensure all code follows PEP 8 standards and passes the following checks:

- Run `black .` for formatting.
- Run `flake8` for linting.
- Run `pytest` for unit tests.

## License

This project is licensed under the MIT License. See `LICENSE` for more information.

## Contact

For questions or support, reach out to `support@omnitensor.io`.