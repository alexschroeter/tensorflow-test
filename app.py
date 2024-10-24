from arkitekt_next import register
import time
import tensorflow as tf

@register
def version() -> str:
    """
    Returns a string with the tensorflow version

    Returns
    -------
    str
        A string with the version of tensorflow
    """
    return f"Tensorflow {tf.__version__}"


@register
def list_gpus() -> str:
    """
    Returns a string of the available physical devices.

    Returns
    -------
    str
        list of physical devices
    """ """"""
    return tf.config.list_physical_devices('GPU') 
