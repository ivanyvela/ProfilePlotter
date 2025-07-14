from setuptools import setup, find_packages

setup(
    name='profileplotter',
    version='0.1.0',
    description='Plot geophysical profiles with TEM, boreholes, and satellite background',
    author='Ivan Vela',
    author_email='ivan.yelamos@gmail.com',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'geopandas',
        'matplotlib',
        'numpy',
        'pandas',
        'rasterio',
        'scipy',
        'shapely',
],
    python_requires='>=3.10, <=3.12',
)


