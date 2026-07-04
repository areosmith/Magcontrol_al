from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import ctypes
import numpy

from artdaq.dmc._lib import lib_importer, wrapped_ndpointer, ctypes_byte_str
class Functions(object):
    def service_enable(bEnable):
        cfunc = lib_importer.windll.DMCSCSL_ServiceEnable
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_uint]
        cfunc.restype = ctypes.c_int32
        return_code = cfunc(bEnable)
        return return_code

    def get_client_port(port):
        cfunc = lib_importer.windll.DMCSCSL_GetClientPort
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.POINTER(ctypes.c_int32)]
        cfunc.restype = ctypes.c_int32
        return_code = cfunc(ctypes.byref(port))
        return return_code

    def update_device_tree(self):
        cfunc = lib_importer.windll.DMCSCSL_UpdateDeviceTree
        if cfunc.argtypes is None:
            pass
        cfunc.restype = ctypes.c_int32
        return_code = cfunc()
        return return_code

    def check_alive(self):
        cfunc = lib_importer.windll.DMCSCSL_CheckAlive
        if cfunc.argtypes is None:
            pass
        cfunc.restype = ctypes.c_int32
        return_code = cfunc()
        return return_code

    def re_allocate_MHWRCSM(self):
        cfunc = lib_importer.windll.DMCSCSL_ReAllocateMHWRCSM
        if cfunc.argtypes is None:
            pass
        cfunc.restype = ctypes.c_int32
        return_code = cfunc()
        return return_code

    def get_devicer_info(device_name, device_type, serial_number, device_status, slot_info):
        cfunc = lib_importer.windll.DMCSCSL_GetDeviceInfo
        if cfunc.argtypes is None:
            cfunc.argtypes = [wrapped_ndpointer(ctypes_byte_str, ctypes.POINTER(ctypes.c_char),
                              ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32))]
        cfunc.restype = ctypes.c_int32
        return_code = cfunc(device_name, ctypes.byref(device_type), ctypes.byref(serial_number), ctypes.byref(device_status), ctypes.byref(slot_info))
        return return_code

    def clear_task_by_name(task_name):
        cfunc = lib_importer.windll.DMCSCSL_ClearTaskByName
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.POINTER(ctypes.c_char)]
        cfunc.restype = ctypes.c_int32
        return_code = cfunc(ctypes.byref(task_name))
        return return_code

    def add_device_to_task(task_name, device_name):
        cfunc = lib_importer.windll.DMCSCSL_AddDeviceToTask
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.POINTER(ctypes.c_char), wrapped_ndpointer(dtype=numpy.char, flags=('C', 'W'))]
        cfunc.restype = ctypes.c_int32
        return_code = cfunc(ctypes.byref(task_name), device_name)
        return return_code

    def reset_device(device_name):
        cfunc = lib_importer.windll.DMCSCSL_ResetDevice
        if cfunc.argtypes is None:
            cfunc.argtypes = [wrapped_ndpointer(dtype=numpy.char, flags=('C', 'W'))]
        cfunc.restype = ctypes.c_int32
        return_code = cfunc(device_name)
        return return_code

    def get_device_list(dev_class, buffer, buffer_size, dev_count):
        cfunc = lib_importer.windll.DMCSCSL_GetDeviceList
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_uint32, ctypes.c_void_p,
                              ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
        cfunc.restype = ctypes.c_int32
        return_code = cfunc(dev_class, buffer, buffer_size, ctypes.byref(dev_count))
        return return_code

    def add_device(device_name, BusType, Category, DevType, SerialNumber, DeviceInstanceId, DevStatus, slotInfo):
        cfunc = lib_importer.windll.DMCSCSL_AddDevice
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes_byte_str, ctypes_byte_str, ctypes_byte_str,
                              ctypes_byte_str,  ctypes.c_uint32,
                              ctypes_byte_str, ctypes.c_uint32, ctypes.c_uint32]
        cfunc.restype = ctypes.c_int32
        return_code = cfunc(device_name, BusType, Category, DevType, SerialNumber, DeviceInstanceId, DevStatus, slotInfo)
        return return_code

    def remove_device(device_name):
        cfunc = lib_importer.windll.DMCSCSL_RemoveDevice
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes_byte_str]
        cfunc.restype = ctypes.c_int32
        return_code = cfunc(device_name)
        return return_code

    def rename_device(device_name, new_name):
        cfunc = lib_importer.windll.DMCSCSL_RenameDevice
        if cfunc.argtypes is None:
             cfunc.argtypes = [ctypes_byte_str, ctypes_byte_str]

        cfunc.restype = ctypes.c_int32
        return_code = cfunc(device_name, new_name)
        return return_code

    def reset_device_list(self):
        cfunc = lib_importer.windll.DMCSCSL_ResetDeviceList
        if cfunc.argtypes is None:
            pass
        cfunc.restype = ctypes.c_int32
        return_code = cfunc()
        return return_code

