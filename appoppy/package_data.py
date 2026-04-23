
import os
from pathlib import Path
ROOT_DIR_KEY = 'APPOPPY_ROOT_DIR'



def set_data_root_dir(folder):
    """
    Set the root directory for measurement data.
    
    Parameters
    ----------
    folder : str or Path
        Path to the root directory
    
    Examples
    --------
    >>> from appoppy.package_data import set_data_root_dir
    >>> set_data_root_dir('/path/to/data')
    """
    os.environ[ROOT_DIR_KEY] = str(folder)

def _data_root_dir_legacy():
    import pkg_resources

    dataroot = pkg_resources.resource_filename(
        'appoppy',
        'data')
    return Path(dataroot)

def _data_root_dir_default():
    try:
        import importlib.resources
        dataroot = importlib.resources.files('appoppy') / 'data'
    except AttributeError:
        dataroot = _data_root_dir_legacy()
    return dataroot

def data_root_dir():
    try:
        return Path(os.environ[ROOT_DIR_KEY])
    except KeyError:
        return _data_root_dir_default()
    
