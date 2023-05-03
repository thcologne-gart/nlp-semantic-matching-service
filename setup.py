import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nlp_semantic_matching_service",
    version="0.0.1",
    author="Maximilian Both",
    description="A semantic matching service implementing NLP matching",
    long_description=long_description,
    long_description_context_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires=">=3.9"
)
