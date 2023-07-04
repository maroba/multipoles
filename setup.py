from setuptools import setup

setup(
    name='multipoles',
    version='0.3.2',
    description='A Python package for multipole expansions of electrostatic or gravitational potentials',
    long_description="""*multipoles* is a Python package for multipole expansions of the solutions of the Poisson equation     
       (e.g. electrostatic or gravitational potentials). It can handle discrete and continuous charge or mass distributions.
    """,

    license='MIT',
    url='https://github.com/maroba/multipoles',

    author='Matthias Baer',
    author_email='matthias.r.baer@googlemail.com',

    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords=['multipole expansion',  'physics', 'scientific-computing'],
    packages=['multipoles'],
    install_requires=['numpy', 'scipy'],

)
