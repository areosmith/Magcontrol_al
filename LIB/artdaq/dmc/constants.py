from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from enum import Enum
import ctypes
from ctypes import *
# Constants
DEV_CLASS_DAQ = 1 << 0
DEV_CLASS_SCOPE = 1 << 1
DEV_CLASS_SWITCH = 1 << 2
DEV_CLASS_DMM = 1 << 3
DEV_CLASS_ALL = 0xFFFFFFFF

class DevBusType(Enum):
   USB = 1 << 0
   PCI = 1 << 1
   TCPIP = 1 << 2

class DevState(Enum):
    DN_ROOT_ENUMERATED = 0x00000001  # Was enumerated by ROOT
    DN_DRIVER_LOADED = 0x00000002  # Has Register_Device_Driver
    DN_ENUM_LOADED = 0x00000004  # Has Register_Enumerator
    DN_STARTED = 0x00000008  # Is currently configured
    DN_MANUAL = 0x00000010  # Manually installed
    DN_NEED_TO_ENUM = 0x00000020  # May need reenumeration
    DN_NOT_FIRST_TIME = 0x00000040  # Has  received a config
    DN_HARDWARE_ENUM = 0x00000080  # Enum generates hardware ID
    DN_LIAR = 0x00000100  # Lied about can reconfig once
    DN_HAS_MARK = 0x00000200  # Not CM_Create_DevInst lately
    DN_HAS_PROBLEM = 0x00000400  # Need device installer
    DN_FILTERED = 0x00000800  # Is filtered
    DN_MOVED = 0x00001000  # Has been moved
    DN_DISABLEABLE = 0x00002000  # Can be disabled
    DN_REMOVABLE = 0x00004000  # Can be removed
    DN_PRIVATE_PROBLEM = 0x00008000  # Has a private problem
    DN_MF_PARENT = 0x00010000  # Multi function parent
    DN_MF_CHILD = 0x00020000  # Multi function child
    DN_WILL_BE_REMOVED = 0x00040000  # DevInst is beingremoved

    DEVICE_ID_MAX = 128
    TASK_NAMELEN_MAX = 64
    DEV_NAMELEN_MAx = 64
    DEV_TYPELEN_MAX = 16
    DEV_CATLEN_MAX = 16
    ALIAS_MAX = 64

class Error(Enum):
    DMCSCSL_SUCCESS = 0
    DMCSCSL_ERR_PALMemoryConfigurationFault = -50350
    DMCSCSL_ERR_PALDeviceNotSupported = -50302
    DMCSCSL_ERR_PALDeviceUnknown = -50301
    DMCSCSL_ERR_SerivesCanNotConnect = -229779
    DMCSCSL_ERR_SendSerivesData = -229778
    DMCSCSL_ERR_RecieveSerivesData = -229776
    DMCSCSL_ERR_SerivesResponseError = -229775
    DMCSCSL_ERR_DuplicateTask = -200089
    DMCSCSL_ERR_DeviceNotExist = -200220
    DMCSCSL_ERR_FileNotFound = -26103
    DMCSCSL_ERR_FileCannotAccess = -26104
    DMCSCSL_ERR_DeviceAlreadyExist = -26105
    DMCSCSL_ERR_ReadBufferTooSmall = -200229
    DMCSCSL_ERR_NULLPtr = -200604
    DMCSCSL_ERR_InvalidTaskName = -201340
    DMCSCSL_ERR_InvalidDeviceName = -201339
    DMCSCSL_ERR_InvalidBusType = -201338
    DMCSCSL_ERR_InvalidCategory = -201337

class Dev_Node(Structure):
    _fields_ = [("DeviceName", ctypes.c_char*64),
                ("DeviceType", ctypes.c_char*16),
                ("DeviceInstanceId", ctypes.c_char * 128),
                ("DeviceCategory", ctypes.c_char * 16),
                ("DeviceClass", ctypes.c_uint32),
                ("BusType", ctypes.c_uint32),
                ("SerialNumber", ctypes.c_uint32),
                ("slotInfo", ctypes.c_uint32),
                ("DevNodeStatus", ctypes.c_uint32)]





