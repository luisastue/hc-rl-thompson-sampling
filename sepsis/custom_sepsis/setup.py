from setuptools import setup, find_packages

setup(
    name="custom_sepsis",         # Name of your package
    version="0.1.0",              # Version number
    packages=find_packages("src"),  # Look for packages in the "src" directory
    # Define "src" as the root directory for packages
    package_dir={"": "src"},
    install_requires=[            # Add dependencies if necessary
        # Example dependency (use "gymnasium" instead of "gym" if you're using its newer version)
        "gymnasium",
    ],
)
