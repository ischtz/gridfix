from setuptools import setup

setup(name='gridfix',
      version='0.3',
      description='Parcellation-based fixation preprocessing for GLMM',
      url='https://github.com/ischtz/gridfix',
      author='Immo Schuetz',
      author_email='ischtz@posteo.eu',
      license='MIT',
      packages=['gridfix'],
      install_requires=['numpy', 'matplotlib', 'scipy', 'pandas'],
      zip_safe=True)