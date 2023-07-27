from setuptools import find_packages, setup
from typing import List


HYP_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    """
    returns list of requirements
    """
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYP_E_DOT in requirements:
            requirements.remove(HYP_E_DOT)

    return requirements


setup(
name = 'gen_ML',
version='0.0.1',
author='abhi',
author_email='as711@uw.edu',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')


)