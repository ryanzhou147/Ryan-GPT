import importlib.metadata

try:
	__version__ = importlib.metadata.version("Ryan-GPT-systems")
except importlib.metadata.PackageNotFoundError:
	try:
		__version__ = importlib.metadata.version("ryan-gpt-systems")
	except importlib.metadata.PackageNotFoundError:
		__version__ = "0.0.0"