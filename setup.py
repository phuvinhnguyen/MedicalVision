from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="MedicalVision",
    version="0.1.0",
    description=long_description,
    packages=find_packages(),  # Automatically find packages in your project
    install_requires=[
        "torch>=1.9.0",  # PyTorch for deep learning
        "torchvision>=0.10.0",  # torchvision for image processing utilities
        "Pillow>=8.2.0",  # Image processing library
        "numpy>=1.19.2",  # Array processing library
        "matplotlib>=3.3.4",  # Plotting library
        "opencv-python>=4.5.1.48",  # Computer vision library
        "lightning>=2.3.3",
        "coco-eval",
        "funcy",
    ],
    package_data={
        'MedicalVision': ['preprocess/*'],
    },
    include_package_data=True,  # Include data files specified in package_data
)