import yaml
import json
from types import SimpleNamespace

def dict2obj(d):
    # checking whether object d is a
    # instance of class list
    if isinstance(d, list):
           d = [dict2obj(x) for x in d]
 
    # if d is not a instance of dict then
    # directly object is returned
    if not isinstance(d, dict):
           return d
  
    # declaring a class
    class C:
        pass
  
    # constructor of the class passed to obj
    obj = C()
  
    for k in d:
        obj.__dict__[k] = dict2obj(d[k])
  
    return obj

def dict_merge(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], dict)):  #noqa
        # if (k in dct and isinstance(dct[k], dict) ):    
            dict_merge(dct[k], merge_dct[k])
        else:
            if k in dct.keys():
                dct[k] = merge_dct[k]

class Config(object):
    """
    Simple dict wrapper that adds a thin API allowing for slash-based retrieval of
    nested elements, e.g. cfg.get_config("meta/dataset_name")
    """
    def __init__(self, default_path, config_path=None) -> None:
        
        cfg = {}
        if config_path is not None:
            with open(config_path) as cf_file:
                cfg = yaml.safe_load( cf_file.read())     
        
        with open(default_path) as def_cf_file:
            default_cfg = yaml.safe_load( def_cf_file.read())

        dict_merge(default_cfg, cfg)

        self._data = default_cfg
    
    def get(self, path=None, default=None):

        sub_dict = dict(self._data)
        if path is None:
            return sub_dict

        path_items = path.split("/")[:-1]
        data_item = path.split("/")[-1]

        try:
            for path_item in path_items:
                sub_dict = sub_dict.get(path_item)

            value = sub_dict.get(data_item, default)

            return value
        except (TypeError, AttributeError):
            return default

class ConfigObj():
    def __init__(self, default_path, config_path=None) -> None:
        cfg = {}
        if config_path is not None:
            with open(config_path) as cf_file:
                cfg = yaml.safe_load( cf_file.read())     
        
        with open(default_path) as def_cf_file:
            default_cfg = yaml.safe_load( def_cf_file.read())

        dict_merge(default_cfg, cfg)
        self._data_obj = json.loads(json.dumps(default_cfg), object_hook=lambda item: SimpleNamespace(**item))
    def get(self):
        return self._data_obj
    def __str__(self):
        return str(self._data)

def config_parser(default_path, config_path=None, return_obj=True):
    if return_obj:
        return ConfigObj(default_path=default_path, config_path=config_path)
    return Config(default_path=default_path, config_path=config_path)


if __name__ == "__main__":
    config = config_parser(default_path='./config/fmgssl_cifar10_4000_default.yaml', config_path='config/fmgssl_test.yaml')
    data = config.get()
    print(data)