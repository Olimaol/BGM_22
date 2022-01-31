# setuptools
try:
    import setuptools
    print('Checking for setuptools... OK')
except:
    print('Checking for setuptools... NO')
    print('Error : Python package "setuptools" is required.')
    exit(0)


# ANNarchy
try:
    print('Checking for ANNarchy... ', end='')
    import ANNarchy
except:
    print('NO')
    print('Error : Python package "ANNarchy" is required.')
    print('For installation check: https://annarchy.readthedocs.io/en/latest/')
    exit(0)

dependencies = [
    'numpy',
    'scipy',
    'matplotlib',
    'cython',
    'sympy',
    'hyperopt',
    'ANNarchy'
]

setuptools.setup(
    name="BGM_22",
    version="0.0.1",
    description="A basal ganglia model in ANNarchy.",
    url="https://github.com/Olimaol/BGM_22",
    packages=setuptools.find_packages(),
    install_requires=dependencies
)
