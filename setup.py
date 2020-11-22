import setuptools

with open('README.md', 'r') as rf:
    readme = rf.read()

setuptools.setup(
    name='gridfix',
    version='0.3.1',
    description='Parcellation-based gaze fixation preprocessing for GLMM analysis',
    long_description=readme,
    long_description_content_type='text/markdown',
    url='https://github.com/ischtz/gridfix',
    author='Immo Schuetz',
    author_email='schuetz.immo@gmail.com',
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'matplotlib', 'scipy', 'pandas'],
    zip_safe=True,
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3'
    ],
    python_requires='>=3.6')

