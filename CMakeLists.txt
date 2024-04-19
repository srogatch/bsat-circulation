cmake_minimum_required(VERSION 3.25.2 FATAL_ERROR)
project(sparse-tiling LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(TBB REQUIRED COMPONENTS tbb)

include(CTest)

set(CMAKE_CUDA_ARCHITECTURES
  80
  86
  89
  90
  CACHE STRING "CUDA compute capabilities to target for" FORCE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --maxrregcount 128") # 
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC --use_fast_math --ptxas-options=-v")
string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
if(uppercase_CMAKE_BUILD_TYPE MATCHES DEBUG)
  set(REVEAL_SYMBOLS ON)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -O0")
elseif(uppercase_CMAKE_BUILD_TYPE MATCHES RELWITHDEBINFO)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -O2")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O2")
else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -Xptxas -O3")
endif()

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Werror all-warnings")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --nvlink-options --suppress-stack-size-warning")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda -Xcompiler -fopenmp")

# lots of warnings and all warnings as errors
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall -Xcompiler -Wextra \
   -Xcompiler -mrdrnd") # -Xcompiler -mbmi -Xcompiler -mbmi2 -Xcompiler -mavx2 -Xcompiler -mfma -Xcompiler -mavx
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Werror -mrdrnd -Wno-write-strings") # -mavx2 -mfma -mbmi -mbmi2 -mavx -flto
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -pedantic -Werror -mrdrnd -Wno-write-strings") # -mavx2 -mfma -mbmi -mbmi2  -mavx -flto
  # if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  #   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -stdlib=libc++")
  #   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
  # endif()
elseif(MSVC)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /W4 -Xcompiler /arch:AVX2 \
    -Xcompiler /GL -Xcompiler /Qvec -Xcompiler /Oi -Xcompiler /Ob3 -Xcompiler /WX")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /arch:AVX2 /GL /Qvec /Oi /Ob3 /WX")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W4 /arch:AVX2 /GL /Qvec /Oi /Ob3 /WX")
else()
  message(FATAL_ERROR "Unsupported compiler")
endif()

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(RogasatGPU
  gpu/BSAT-GPU.cu gpu/GpuLinkage.cu gpu/CpuInit.cpp gpu/Common.cpp)
target_include_directories(RogasatGPU PRIVATE gpu)
target_link_libraries(RogasatGPU PRIVATE CUDA::cudart OpenMP::OpenMP_CXX tbb)
