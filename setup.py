from setuptools import find_packages,setup
from typing import List


# -e . is added to requirements.txt so it could initiate setup.py
# as part of below function this -e . should not be listed.
HYPHEN_E = '-e .' # constant create to check 


def list_requirements(filepath:str)->List[str]:
    """
    This functions lists out all the requirements under the requirements.txt file. 
    """
    with open(filepath,'rt') as f:
        requirements = f.readlines()
        requirements = [line.replace("\n","") for line in requirements]
        if HYPHEN_E in requirements:
            requirements.remove(HYPHEN_E)
    return requirements

setup(
    name='End_To_End_ML_Project',
    version='0.1',
    author='Sreenivas Pakyala',
    author_email='pakyalasreenivas@gmail.com',
    packages=find_packages(),
    keywords='MachineLearning',
    # requires=list_requirements('requirements.txt'),
    install_requires=list_requirements('requirements.txt'),
)

# if __name__ == '__main__':
#     print(list_requirements('requirements.txt'))
#     print('All requirements are printed out...')