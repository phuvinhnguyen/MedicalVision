from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vision_project",  # Replace with your own project name
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python project for computer vision tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # Replace with your GitHub repo URL
    url="https://github.com/yourusername/vision_project",
    packages=find_packages(),  # Automatically find packages in your project
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.6',
    install_requires=[
        "torch>=1.9.0",  # PyTorch for deep learning
        "torchvision>=0.10.0",  # torchvision for image processing utilities
        "Pillow>=8.2.0",  # Image processing library
        "numpy>=1.19.2",  # Array processing library
        "matplotlib>=3.3.4",  # Plotting library
        "opencv-python>=4.5.1.48",  # Computer vision library
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.2",  # Testing framework
            "black>=20.8b1",  # Code formatter
            "flake8>=3.8.4",  # Linting tool
        ],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "vision_project=vision_project.main:main",  # Replace with your main script
        ],
    },
)
