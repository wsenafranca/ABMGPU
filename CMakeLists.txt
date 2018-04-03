cmake_minimum_required(VERSION 3.9)
project(ABMGPU)

enable_language(CUDA)
find_package(CUDA)
string(APPEND CMAKE_CUDA_FLAGS " -arch=compute_30 -code=sm_30")

set(CUDA_SDK_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR})

set(SRC
	src/AgentSet.cu
	src/RandomBase.cu
	src/RandomFloat.cu
	src/RandomInt.cu
	src/Simulation.cu
	src/SpatialNeighborhood.cu
)

set(INCLUDES
	src/AgentSet.cuh
	src/AgentSetKernels.cuh
	src/Collection.cuh
	src/RandomBase.cuh
	src/RandomFloat.cuh
	src/RandomInt.cuh
	src/Simulation.cuh
	src/Society.cuh
	src/SpatialNeighborhood.cuh
	src/ABMGPU.h
)

option(STATIC "Generate a static lib" OFF)
if(STATIC)
	ADD_LIBRARY(ABMGPU STATIC ${SRC})
else()
	ADD_LIBRARY(ABMGPU SHARED ${SRC})
	file(COPY ${INCLUDES}
		 DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/include)
endif()

set_property(TARGET ABMGPU PROPERTY CUDA_STANDARD 11)

find_cuda_helper_libs(curand)
target_link_libraries(ABMGPU ${CUDA_curand_LIBRARY})

install(TARGETS ABMGPU DESTINATION lib)
install(FILES ${INCLUDES} DESTINATION include)

option(BUILD_DOCUMENTATION "Use Doxygen to create the HTML based API documentation" OFF)
if(BUILD_DOCUMENTATION)
  FIND_PACKAGE(Doxygen)
  if (NOT DOXYGEN_FOUND)
    message(FATAL_ERROR
      "Doxygen is needed to build the documentation. Please install it correctly")
  endif()
  add_custom_target (
  	doc ALL
  	${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile
  	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
endif()
