import json
import os 
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

config_path = './config.json'
config = Struct(**{})

try:
    with open(config_path) as json_file:
        args = json.load(json_file)
        for key in args:
            print(key.upper(), key.upper() in os.environ)
            if key.upper() in os.environ:
                args[key] = os.environ[key.upper()]
        print(args)
        config = Struct(**args)
except:
    os.chdir('../')
    with open(config_path) as json_file:
        args = json.load(json_file)
        for key in args:
            if key.upper() in os.environ:
                args[key] = os.environ[key.upper()]
        config = Struct(**args)



