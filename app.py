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
    physical_devices = tf.config.list_physical_devices()
    result = ""
    for i in range(len(physical_devices)):
        if physical_devices[i].device_type == 'GPU' or physical_devices[i].device_type == 'XPU':
            result += f"GPU {i}: {physical_devices[i].name}\n"
    return result
