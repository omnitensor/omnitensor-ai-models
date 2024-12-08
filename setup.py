from setuptools import setup, find_packages

setup(
    name='omnitensor_ai_models',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch>=1.12.1',
        'transformers>=4.24.0',
        'numpy>=1.21.0',
        'pandas>=1.3.3',
        'scikit-learn>=1.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0',
            'black>=22.3.0',
            'flake8>=4.0',
            'isort>=5.10.0',
            'mypy>=0.942',
        ]
    },
    description="AI models package for OmniTensor project.",
    author="OmniTensor Team",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)