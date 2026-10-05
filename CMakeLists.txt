cmake_minimum_required(VERSION 3.20)
project(ImgReconstruct_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui photo dnn)
find_package(OpenMP REQUIRED)
if(NOT WIN32)
    # non-Windows crypto backend (Windows uses CNG / bcrypt)
    find_package(OpenSSL REQUIRED)
endif()

# contrib quality module (SSIM) -- optional; quality_utils falls back to
# reporting SSIM as unavailable when the header is missing
set(CS_HAVE_QUALITY OFF)
foreach(_inc ${OpenCV_INCLUDE_DIRS})
    if(EXISTS "${_inc}/opencv2/quality.hpp")
        set(CS_HAVE_QUALITY ON)
    endif()
endforeach()
if(CS_HAVE_QUALITY)
    list(APPEND OpenCV_LIBS opencv_quality)
    message(STATUS "OpenCV quality module found: SSIM enabled")
else()
    message(STATUS "OpenCV quality module not found: SSIM will report n/a")
endif()

# core library shared by the CLI and the test suite
add_library(cs_core STATIC
    ImgReconstruct_backend/CS_encryption.cpp
    ImgReconstruct_backend/clip_scorer.cpp
    ImgReconstruct_backend/crypto_utils.cpp
    ImgReconstruct_backend/helper_functions.cpp
    ImgReconstruct_backend/image_decryption.cpp
    ImgReconstruct_backend/image_encryption.cpp
)
target_include_directories(cs_core PUBLIC ImgReconstruct_backend)
if(WIN32)
    # ONNX Runtime (CLIP scorer); headers + import lib vendored under third_party.
    # Delay-loaded so the CLI still starts when onnxruntime.dll is absent;
    # the scorer then reports a clean error instead of a loader failure.
    target_include_directories(cs_core PRIVATE ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/include)
    target_link_libraries(cs_core PUBLIC ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib/onnxruntime.lib)
    if(MSVC)
        target_link_options(cs_core PUBLIC /DELAYLOAD:onnxruntime.dll)
        target_link_libraries(cs_core PUBLIC delayimp.lib)
    endif()
endif()
# quality_utils.hpp lives next to the core sources; tests include it from there
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
