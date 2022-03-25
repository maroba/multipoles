from setuptools import setup

setup(
    name='multipoles',
    version='0.2.0',
    description='A Python package for multipole expansions of electrostatic or gravitational potentials',
    long_description="""*multipoles* is a Python package for multipole expansions of the solutions of the Poisson equation     
       (e.g. electrostatic or gravitational potentials). It can handle discrete and continuous charge or mass distributions.
    """,

    license='MIT',
    url='https://github.com/maroba/multipoles',

    author='Matthias Baer',  # Optional
    author_email='mrbaer@t-online.de',  # Optional

    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Mathematics',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords=['multipole expansion',  'physics', 'scientific-computing'],  # Optional
    packages=['multipoles'],
    install_requires=['numpy', 'scipy'],  # Optional

)
