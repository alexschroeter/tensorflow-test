from arkitekt_next import register
import time
import tensorflow as tf

try:
    import intel_extension_for_tensorflow as tf
except:
    "Intel extension for tensorflow not found"

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
    result = ""
    for i in range(tf.config.list_physical_devices()):
        if i.device_type == 'GPU' or i.device_type == 'XPU':
            result += f"GPU {i}: {i.name}\n"
    return result
