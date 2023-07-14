from setuptools import setup, find_packages

setup(
    name='anthropic_with_functions',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'anthropic',
    ],
    author='Matt Shumer',
    author_email='mattshumertech@gmail.com',
    description='A library to use the Anthropic Claude models with OpenAI-like Functions.',
)