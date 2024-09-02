from setuptools import setup, find_packages

# meta data of the package 
NAME = 'Insurance premium prediction'
DESCRIPTION = 'Loan prediction model'
URL ='https://github.com/Danlof/medical_insurance'
EMAIL ='mdanlof@gmail.com'
AUTHOR = 'Danlof Musyoki'
REQUIRES_PYTHON = '>=3.10.12'

setup(
    name='Insurance_Premium_Prediction',
    version='0.1',
    packages=find_packages(),
     install_requires=[
        # List of dependencies, if any
        'pandas',
        'numpy',
        'scikit-learn',
        'seaborn',
        'scipy',
        'xgboost',
        'matplotlib',
        'joblib',
        'gunicorn',
    ],
)
