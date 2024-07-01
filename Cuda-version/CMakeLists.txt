###########################################################################
# CUCSX - Berenger Bramas Inria - 2023
# Under MIT Licence, please you must read the LICENCE file.
###########################################################################
cmake_minimum_required (VERSION 3.20 FATAL_ERROR)

set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

#===========================================================================
# Project Declaration
#===========================================================================
project(CUCSX LANGUAGES CXX VERSION 1.0.0)
set(CUCSX_SOURCE_DIR ${CUCSX_SOURCE_DIR} )
set(CUCSX_BINARY_DIR ${CUCSX_BINARY_DIR} )


include(CheckLanguage)
check_language(CUDA)

enable_language(CUDA)
if(NOT DEFINED CUDA_SM)
	MESSAGE(FATAL_ERROR "You must set CUDA target architecture, like 75 or 80, in CUDA_SM")
endif()
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_SM} CACHE STRING "SM ARCH" FORCE)
MESSAGE(STATUS "CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")

# check if compiling into source directories
string(COMPARE EQUAL "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}" insource)
if(insource)
  message(FATAL_ERROR "${PROJECT_NAME} requires an out of source build.\
                      Remove CMakeCache.txt (alternatively you can remove all untracked files with\
                      git clean -i, git clean -d or git clean -df). Then, create a subirectory outside\
                      the CUCSX source tree, cd into the directory you just created and then type\
                      cmake <path to CUCSX repository>.\
                      Alternatively you can create a build directory at the root of the CUCSX\
                      source tree but then it is recommended to name it 'build' as it is the only\
                      build directory name that is explicitly marked as to be ignored by Git.")
endif(insource)

set(CUCSX_LIBRARIES "cudart;cublas")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CUDA_VERBOSE_BUILD ON)
include(FindCUDA)
include_directories(${CUDA_INCLUDE_DIRS})
list(APPEND CUCSX_LIBRARIES ${CUDA_LIBRARIES})

if(CMAKE_BUILD_TYPE STREQUAL "DEBUG" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
endif()
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -DSTANDALONE" CACHE STRING "Flags for NVCC")   
set(CMAKE_CUDA_FLAGS_RELEASE "-DSTANDALONE" CACHE STRING "Flags for NVCC")    
    
INCLUDE_DIRECTORIES(
     ${CMAKE_CURRENT_SOURCE_DIR}
     ${CMAKE_CURRENT_BINARY_DIR}
)
        
# Release by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RELEASE" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo" FORCE)
endif()

MESSAGE(STATUS "Main -- CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE}")

option(NOMODOPT "Modulo optim" OFF)

# Extra-flags
SET( CUCSX_EXTRA_CXX_FLAGS "-funroll-loops" CACHE STRING "These flags are used for all configurations"  )
set(CUCSX_CXX_FLAGS  "-std=c++17 -DSTANDALONE ${CUCSX_CXX_FLAGS} ${CUCSX_EXTRA_CXX_FLAGS}")

SET( CUCSX_EXTRA_CXX_FLAGS_RELEASE "-O3" CACHE STRING "These flags are added to the usual ones for RELEASE"  )
set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} ${CUCSX_EXTRA_CXX_FLAGS_RELEASE}")

SET( CUCSX_EXTRA_CXX_FLAGS_DEBUG "-O0" CACHE STRING "These flags are added to the usual ones for DEBUG"  )
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} ${CUCSX_EXTRA_CXX_FLAGS_DEBUG}")

set(CUCSX_CXX_FLAGS "${CUCSX_CXX_FLAGS} -fpic -Wextra -Wnon-virtual-dtor -Wshadow -Wpointer-arith -Wcast-qual -Wall -pedantic -Woverloaded-virtual -Wpointer-arith -Wcast-qual -Wno-error")
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
else()
	set(CUCSX_CXX_FLAGS "${CUCSX_CXX_FLAGS} -Wconversion -Wno-sign-conversion")
endif()


file(GLOB	
    source_tests_files
    ./*.cu
    )


# Adding the project sources dir as an include dir
INCLUDE_DIRECTORIES( ${UTESTS_CUCSX_SOURCE_DIR})

MESSAGE(STATUS " -- CUCSX_LIBRARIES ${CUCSX_LIBRARIES}")

# Add execs - 1 cpp = 1 exec
foreach(exec ${source_tests_files}) 
    get_filename_component(
	    execname ${exec}
	    NAME_WE
    )
	MESSAGE(STATUS " -- Add ${execname}")

	set_source_files_properties(${exec} PROPERTIES LANGUAGE CUDA)
	add_executable(
			    ${execname}
			    ${exec}
		    )

	target_link_libraries(
	    ${execname}
	    ${CUCSX_LIBRARIES}
	    )
	target_compile_options(${execname} PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-lineinfo --generate-line-info>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-lineinfo --generate-line-info>
    )
endforeach(exec)


