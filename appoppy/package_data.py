
import os
ROOT_DIR_KEY = 'APPOPPY_ROOT_DIR'

def data_root_dir():

    try:
        return os.environ[ROOT_DIR_KEY]
    except KeyError:
        import pkg_resources
        dataroot = pkg_resources.resource_filename(
            'appoppy',
            'data')
        return dataroot
