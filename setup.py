from pathlib import Path
from setuptools import setup
from setuptools import find_packages

def normalized(name: str) -> str:
    return name.replace("-", "_")

PROJECT = "flax-extra"
PROJECT_DIR = f"src/{normalized(PROJECT)}"
REPOSITORY = f"manifest/{PROJECT}"
README = (Path(__file__).parent / "README.md").read_text()

# Setup project version.
__version__ = None
with open(f"{normalized(PROJECT_DIR)}/version.py") as file:
    exec(file.read(), globals())

# Setup keywords.
# https://setuptools.readthedocs.io/en/latest/references/keywords.html
setup(
    name=PROJECT,
    version=__version__,
    author="Andrei Nesterov",
    author_email="ae.nesterov@gmail.com",
    url=f"https://github.com/{REPOSITORY}",
    description="The package provides extra flexibility to Flax using ideas originated at Trax",
    long_description=README,
    long_description_content_type="text/markdown",
    project_urls={
        "Documentation": f"https://{PROJECT}.readthedocs.io",
        "Source Code": f"https://github.com/{REPOSITORY}",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Typing :: Typed",
    ],
    # Required for mypy to find the installed package.
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={normalized(PROJECT): ["py.typed"]},
    zip_safe=False,
    python_requires=">=3.9",
    install_requires=[
        # Uncomment for local development.
        # f"redex @ file://localhost//{Path('../redex').resolve()}#egg=redex",
        "redex>=0.1",
        "flax>=0.3",
        # Support new JAX custom PRNG key system.
        # "flax @ git+ssh://git@github.com/google/flax@6444f97ef2b363dbef9c834f093516f7e99bb05c#egg=flax",
        "einops>=0.3",
        # Required by `flax.training.checkpoints`
        "tensorflow>=2.6",
    ],
    extras_require={
        "docs": ["sphinx", "furo", "nbsphinx", "ipykernel"],
        "development": ["numpy", "pylint==2.10"],
    },
)
