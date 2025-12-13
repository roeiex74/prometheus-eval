"""
Prometheus-Eval Setup Configuration
A comprehensive framework for rigorous evaluation of LLM prompt effectiveness.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
try:
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "Comprehensive framework for rigorous evaluation of LLM prompt effectiveness"

# Parse requirements.txt
requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
requirements = []
try:
    with open(requirements_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith("#"):
                requirements.append(line)
except FileNotFoundError:
    requirements = []

setup(
    name="prometheus-eval",
    version="0.1.0",
    author="Research Team",
    author_email="research@example.com",
    description="Comprehensive framework for rigorous evaluation of LLM prompt effectiveness",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/prometheus-eval",
    packages=find_packages(where=".", exclude=["tests*", "docs*", "docker-images*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "pytest-timeout>=2.2.0",
            "black>=23.12.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # CLI commands can be added here in the future
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
