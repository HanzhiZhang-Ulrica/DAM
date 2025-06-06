from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dam-attention",
    version="0.1.0",
    author="Hanzhi Zhang, Heng Fan, Kewei Sha, Yan Huang, Yunhe Feng",
    author_email="hanzhi.zhang@unt.edu",
    description="Dynamic Attention Mask for Long-Context LLM Inference Acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HanzhiZhang-Ulrica/DAM",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13.0",
        "transformers>=4.30.0",
        "datasets",
        "triton>=2.0.0",
        "numpy",
        "matplotlib",
        "seaborn",
        "tqdm",
        "scikit-learn",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "mypy",
        ],
    },
) 