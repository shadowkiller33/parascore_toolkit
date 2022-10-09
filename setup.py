from distutils.core import  setup
import setuptools
packages = ['parascore']# 唯一的包名，自己取名
setup(name='parascore',
	version='1.0',
	author='lingfengshen',
    packages=packages,
    package_dir={'requests': 'requests'},)
