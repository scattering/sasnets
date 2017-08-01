from setuptools import setup

setup(
    name='sasnets',
    version='0.1a',
    description='CNN classifier for SANS data',
    author='Chris Wang',
    author_email='wangch00@icloud.com',
    packages=['sasnets', ],
    install_requires=["matplotlib", "Keras", "tensorflow",
                      "numpy>=1.13.0", "scikit-learn>=0.18.2", "hyperas>=0.4",
                      "bottleneck>=1.2.1", "psycopg2", "mock"],
    extras_require={
        'better_json': ['ruamel.yaml>=0.15'],
        'bumps_fit': ['bumps'],
        'docs': ['Sphinx'],
        'mpl_colour': ['seaborn'],
    }
)
