from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ctypes.util import find_library
from ctypes import CDLL
import platform
import ctypes
from numpy.ctypeslib import ndpointer
import struct
import six
import sys
import threading

class CtypesByteString(object):
    """
    Custom argtype that automatically converts unicode strings to ASCII
    strings in Python 3.
    """
    def from_param(self, param):
        if isinstance(param, six.text_type):
            param = param.encode('ascii')
        return ctypes.c_char_p(param)


ctypes_byte_str = CtypesByteString()

class c_bool32(ctypes.c_uint):
    """
    Specifies a custom ctypes data type to represent 32-bit booleans.
    """

    def _getter(self):
        return bool(ctypes.c_uint.value.__get__(self))

    def _setter(self, val):
        ctypes.c_uint.value.__set__(self, int(val))

    value = property(_getter, _setter)

    del _getter, _setter


class CtypesByteString(object):
    """
    Custom argtype that automatically converts unicode strings to ASCII
    strings in Python 3.
    """
    def from_param(self, param):
        if isinstance(param, six.text_type):
            param = param.encode('ascii')
        return ctypes.c_char_p(param)

ctypes_byte_str = CtypesByteString()

def wrapped_ndpointer(*args, **kwargs):
    """
    Specifies an ndpointer type that wraps numpy.ctypeslib.ndpointer and
    allows a value of None to be passed to an argument of that type.
    """
    if sys.version_info < (3,):
        if 'flags' in kwargs:
            kwargs['flags'] = tuple(
                f.encode('ascii') for f in kwargs['flags'])

    base = ndpointer(*args, **kwargs)

    def from_param(cls, obj):
        if obj is None:
            return obj
        return base.from_param(obj)

    return type(base.__name__, (base,),
                {'from_param': classmethod(from_param)})


def enum_bitfield_to_list(bitfield_value, bitfield_enum_type,
                          actual_enum_type):
    """
    Converts a bitfield value to a list of enums.

    Args:
        bitfield_value (int): Specifies the value of the bitfield.
        bitfield_enum_type (enum.Enum): Specifies the bitfield enum type
            from which to mask and extract the enum values.
        actual_enum_type (enum.Enum): Specifies the actual enum type.
    Returns:
        List[enum.Enum]: Indicates the converted list of enums.
    """
    supported_values = []
    for bitfield_mask in bitfield_enum_type:
        if bitfield_value & bitfield_mask.value:
            enum_value = next(
                e for e in actual_enum_type if e.name == bitfield_mask.name)
            supported_values.append(enum_value)

    return supported_values


def enum_list_to_bitfield(enum_list, bitfield_enum_type):
    """
    Converts a list of enums to a bitfield value.

    Args:
        enum_list (List[enum.Enum]): Specifies the list of enums.
        bitfield_enum_type (enum.Enum): Specifies the bitfield enum type
            from which to mask and extract the enum values.
    Returns:
        int: Indicates the value of the bitfield.
    """
    bitfield_value = 0
    for enum_value in enum_list:
        bitfield_mask = next(
            b for b in bitfield_enum_type if b.name == enum_value.name)
        bitfield_value |= bitfield_mask.value

    return bitfield_value


class DMCLibImporter(object):
    """
    Encapsulates Art_DAQ library importing and handle type parsing logic.
    """

    def __init__(self):
        self._windll = None
        self._cdll = None
        self._cal_handle = None

    @property
    def windll(self):
        if self._cdll is None:
            i = struct.calcsize("P")
            if i == 4:
                self._windll = ctypes.windll.LoadLibrary("DMCSCSL_32.dll")
            else:
                self._windll = ctypes.windll.LoadLibrary("DMCSCSL_64.dll")

        return self._windll

lib_importer = DMCLibImporter()




