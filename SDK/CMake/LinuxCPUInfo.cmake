
#
#  Copyright (c) 2008 - 2023 NVIDIA Corporation.  All rights reserved.
#
#  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
#  rights in and to this software, related documentation and any modifications thereto.
#  Any use, reproduction, disclosure or distribution of this software and related
#  documentation without an express license agreement from NVIDIA Corporation is strictly
#  prohibited.
#
#  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
#  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
#  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
#  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
#  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
#  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
#  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
#  SUCH DAMAGES
#

IF(EXISTS "/proc/cpuinfo")

  FILE(READ /proc/cpuinfo PROC_CPUINFO)

  SET(VENDOR_ID_RX "vendor_id[ \t]*:[ \t]*([a-zA-Z]+)\n")
  STRING(REGEX MATCH "${VENDOR_ID_RX}" VENDOR_ID "${PROC_CPUINFO}")
  STRING(REGEX REPLACE "${VENDOR_ID_RX}" "\\1" VENDOR_ID "${VENDOR_ID}")

  SET(CPU_FAMILY_RX "cpu family[ \t]*:[ \t]*([0-9]+)")
  STRING(REGEX MATCH "${CPU_FAMILY_RX}" CPU_FAMILY "${PROC_CPUINFO}")
  STRING(REGEX REPLACE "${CPU_FAMILY_RX}" "\\1" CPU_FAMILY "${CPU_FAMILY}")

  SET(MODEL_RX "model[ \t]*:[ \t]*([0-9]+)")
  STRING(REGEX MATCH "${MODEL_RX}" MODEL "${PROC_CPUINFO}")
  STRING(REGEX REPLACE "${MODEL_RX}" "\\1" MODEL "${MODEL}")

  SET(FLAGS_RX "flags[ \t]*:[ \t]*([a-zA-Z0-9 _]+)\n")
  STRING(REGEX MATCH "${FLAGS_RX}" FLAGS "${PROC_CPUINFO}")
  STRING(REGEX REPLACE "${FLAGS_RX}" "\\1" FLAGS "${FLAGS}")

  # Debug output.
  IF(LINUX_CPUINFO)
    MESSAGE(STATUS "LinuxCPUInfo.cmake:")
    MESSAGE(STATUS "VENDOR_ID : ${VENDOR_ID}")
    MESSAGE(STATUS "CPU_FAMILY : ${CPU_FAMILY}")
    MESSAGE(STATUS "MODEL : ${MODEL}")
    MESSAGE(STATUS "FLAGS : ${FLAGS}")
  ENDIF(LINUX_CPUINFO)

ENDIF(EXISTS "/proc/cpuinfo")
