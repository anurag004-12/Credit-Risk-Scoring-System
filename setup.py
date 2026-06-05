from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent
requirements_file = here / "requirements.txt"
if requirements_file.exists():
    with requirements_file.open() as f:
        install_requires = [r.strip() for r in f.readlines() if r.strip() and not r.startswith('#')]
else:
    install_requires = []

setup(
    name="credit_risk_scoring_system",
    version="0.1.0",
    description="Credit Risk Scoring System",
    packages=find_packages(exclude=("tests", "notebook")),
    include_package_data=True,
    install_requires=install_requires,
    python_requires=">=3.8",
)
