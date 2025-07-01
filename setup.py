from setuptools import setup, find_packages
setup(
    name="agrofuture",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.26",
        "pandas>=2.2",
        "scikit-learn>=1.4",
        "xgboost>=2.0",
        "lightgbm>=4.3",
        "catboost>=1.2"
    ],
    entry_points={
        "console_scripts": [
            "agrofuture=agrofuture.cli:main",
            "agrofuture-train = agrofuture.model_trainer:main",
            "agrofuture-predict=scripts.generate_predictions:main",
        ]
    }
)
