cmake_minimum_required(VERSION 3.9)
project(ABMGPU)

find_package(CUDA QUIET REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-arch=compute_30 -code=sm_30 -std=c++11")

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
	cuda_ADD_LIBRARY(ABMGPU STATIC ${SRC})
else()
	cuda_ADD_LIBRARY(ABMGPU SHARED ${SRC})
	file(COPY ${INCLUDES}
		 DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/include)
endif()

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
