import os
import importlib

# use the config file from a specified path
def from_file(config_file):
	# config_file = os.path.abspath(os.path.expanduser(config_file))
	if not os.path.exists(config_file):
		raise IOError('The config file does not exist!')

	if config_file.endswith('.py'):
		file_path = config_file[:-3]
		file_path = file_path.split('/')
		file_path = '.'.join(file_path)
		config = importlib.import_module(file_path)
		return config
	else:
		raise IOError('Only the .py config file is supported!')

	
          