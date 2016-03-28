from setuptools import setup, find_packages

setup(
	name ='MarkovMixing',
	version = '0.0.2',
	packages = find_packages(),
	author = 'S. Bordt',
	author_email = 'sbordt@posteo.de',
	description = 'Determine mixing times for Markov chains with ~ 100.000 states.',
	license = 'MIT',
	url = 'https://github.com/sbordt/markovmixing',
	install_requires=[
	"networkx >= 1.9",
	"numpy >= 1.8.2",
	"scipy >= 0.13.3",
	"matplotlib >= 1.3.1",
	"nose >= 1.3.1",
],
)