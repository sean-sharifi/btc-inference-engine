"""Setup file for editable installation compatibility"""
from setuptools import setup, find_packages

setup(
    name="btc-options-onchain-engine",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
