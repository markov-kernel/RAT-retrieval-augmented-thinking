from setuptools import setup, find_packages

setup(
    name="rat",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai",
        "python-dotenv",
        "rich",
        "prompt_toolkit",
        "requests",
    ],
    entry_points={
        'console_scripts': [
            'rat-research=rat.rat_research:main',
        ],
    },
    author="Skirano",
    description="Retrieval Augmented Thinking - Enhanced AI responses through structured reasoning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Doriandarko/RAT-retrieval-augmented-thinking",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 