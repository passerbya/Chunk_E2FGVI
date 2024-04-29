# coding=utf-8
from pathlib import Path
from setuptools import find_packages, setup


NAME = 'e2fgvi'
DESCRIPTION = 'Chunk_E2FGVI'

URL = 'https://github.com/passerbya/Chunk_E2FGVI'
EMAIL = '84305424@qq.com'
AUTHOR = 'wangzh'
REQUIRES_PYTHON = '>=3.7.0'
VERSION = '1.0.4'

HERE = Path(__file__).parent


def load_requirements(name):
    required = [i.strip() for i in open(HERE / name)]
    required = [i for i in required if not i.startswith('#')]
    return required


REQUIRED = load_requirements('requirements.txt')

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages('.'),
    install_requires=REQUIRED,
    include_package_data=True,
    entry_points={
        'console_scripts': ['e2fgvi=e2fgvi.wipe:main_worker'],
    },
    license='Attribution-NonCommercial 4.0',
)