"""Python package setup file."""

import setuptools


# Copy README.md text to long description
with open("README.md", "r", encoding="utf-8") as readme_file:
    long_description = readme_file.read()

# Run setup
setuptools.setup(
    name="jax_transfomations3d",
    version="0.1.0",
    license="Apache-2.0",
    description="JAX compatible 3d transformations.",
    author="Carl Goodrich",
    author_email="carlpgoodrich@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cpgoodri/jax_transformations3d",
    project_urls={
        "Bug Tracker": "https://github.com/cpgoodri/jax_transformations3d/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
