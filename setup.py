import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="toy_nn",
    version="1.0.0",
    author="Yacoub Ahmed",
    author_email="yacoub.ahmedy@gmail.com",
    description="A neural network library written using numpy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yacoubb/toy-nn",
    packages=setuptools.find_packages(where="toy_nn"),
    classifiers=["Programming Language :: Python :: 3", "License :: OSI Approved :: MIT License", "Operating System :: OS Independent"],
    python_requires=">=3.6",
)
