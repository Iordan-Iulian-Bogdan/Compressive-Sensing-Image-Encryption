cmake_minimum_required(VERSION 3.20)
project(ImgReconstruct_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui photo)
find_package(OpenMP REQUIRED)
if(NOT WIN32)
    # non-Windows crypto backend (Windows uses CNG / bcrypt)
    find_package(OpenSSL REQUIRED)
endif()

# core library shared by the CLI and the test suite
add_library(cs_core STATIC
    ImgReconstruct_backend/CS_encryption.cpp
    ImgReconstruct_backend/crypto_utils.cpp
    ImgReconstruct_backend/helper_functions.cpp
    ImgReconstruct_backend/image_decryption.cpp
    ImgReconstruct_backend/image_encryption.cpp
    ImgReconstruct_backend/lbfgs.cpp
)
target_include_directories(cs_core PUBLIC ImgReconstruct_backend)
target_link_libraries(cs_core PUBLIC ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
if(NOT WIN32)
    target_link_libraries(cs_core PUBLIC OpenSSL::Crypto)
endif()
if(MSVC)
    target_compile_options(cs_core PRIVATE /arch:AVX2)
else()
    # AVX2 + FMA: the gather/scatter kernels use _mm256_fmadd_ps
    target_compile_options(cs_core PRIVATE -mavx2 -mfma)
endif()

add_executable(ImgReconstruct_backend ImgReconstruct_backend/ImgReconstruct_backend.cpp)
target_link_libraries(ImgReconstruct_backend PRIVATE cs_core)

enable_testing()
add_executable(cs_tests tests/test_main.cpp)
target_link_libraries(cs_tests PRIVATE cs_core)
add_test(NAME cs_tests COMMAND cs_tests)
