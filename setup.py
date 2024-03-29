from setuptools import setup

setup(
    name='eatBoost',
    version='0.1',
    description='Efficiency Analysis Trees Technique',
    url='https://doi.org/10.1016/j.eswa.2020.113783',
    author='Miriam Esteve',
    author_email='miriam.estevec@umh.es',
    packages=['eatBoost'],
    install_requires=['numpy', 'pandas', 'graphviz', 'docplex', "matplotlib"],
    license='AFL-3.0',
    zip_safe=False
)