from setuptools import setup, find_packages
import os

# Funktion zum Einlesen der requirements.txt
def read_requirements():
    with open('requirements.txt', 'r') as file:
        return file.read().splitlines()

setup(
    name="GAN_Implementations",  # Name des Pakets
    version="0.1.0",  # Version deines Pakets
    packages=find_packages(),  # Automatische Paketfindung
    install_requires=read_requirements(),  # Abhängigkeiten aus der requirements.txt
    author="Nicolas Becker",
    author_email="nicolas1998.becker@gmail.com",
    description="Implementation of various GANs and helper functions to define training and custom loaders in Pytorch",
    url="https://github.com/NiggoNiggo/GANs/tree/main",  # URL deines GitHub-Repos
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Python-Versionen, die unterstützt werden
)
