import json
import os 
from pprint import pprint
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

config_path = './config.json'
config = Struct(**{})

try:
    with open(config_path) as json_file:
        args = json.load(json_file)
        for key in args:
            if key.upper() in os.environ:
                if key == 'threshold':
                    args[key] = float(os.environ[key.upper()])
                else:
                    args[key] = os.environ[key.upper()]
        config = Struct(**args)
        print("Config:")
        pprint(args)
except:
    os.chdir('../')
    with open(config_path) as json_file:
        args = json.load(json_file)
        for key in args:
            if key == 'threshold':
                args[key] = float(os.environ[key.upper()])
            else:
                args[key] = os.environ[key.upper()]
        config = Struct(**args)



