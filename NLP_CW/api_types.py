from typing import List, Dict, cast, Callable, Union

class GlobalConfig:
    def __init__(self, **entries): 
        self.__dict__.update(entries)
    