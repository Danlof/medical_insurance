## Debugging 
`find . -name "__pycache__" -exec rm -rf {} +` to remove any pycache present

- If you plan to frequently run scripts from different locations, it might be worth creating a setup.py in the root of your project. You can then install the package in "editable" mode, which makes it easier to import modules across your project.

```
from setuptools import setup, find_packages

setup(
    name='prediction_model',
    version='0.1',
    packages=find_packages(),  # Automatically find all packages in your project
    install_requires=[
        # List of dependencies, if any
        'pandas',
        'numpy',
        'scikit-learn',
        # Add any other dependencies
    ],
)

```

then run this `pip install -e .`