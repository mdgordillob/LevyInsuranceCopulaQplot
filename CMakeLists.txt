cmake_minimum_required(VERSION 3.12)
project(copula_calculations LANGUAGES CXX)

# Find Python interpreter and libraries
# Use Python3::Interpreter and Python3::Development::Embed for modern CMake
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find pybind11
# Method 1: Find installed pybind11 (recommended if installed via pip)
find_package(pybind11 CONFIG REQUIRED)
# Method 2: If pybind11 headers are in a known location (less common)
# add_subdirectory(path/to/pybind11) # If pybind11 is included as a submodule/source

# Find Boost headers (rely on system include paths)
# We installed libboost-math-dev, headers should be findable.
# Linking will happen later.
find_package(Boost REQUIRED) # Find general Boost installation
include_directories(${Boost_INCLUDE_DIRS})

# Add compiler flags for optimization and OpenMP
# Use target_compile_options for modern CMake
set(CMAKE_CXX_STANDARD 17) # Use C++17 standard
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Create the Python extension module
# pybind11_add_module(<module_name> <sources...>)
pybind11_add_module(copula_calculations copula_calculations.cpp)

# Apply compiler options to the target
target_compile_options(copula_calculations PRIVATE -O3 -DNDEBUG)

# Enable OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(copula_calculations PRIVATE OpenMP::OpenMP_CXX)

# Link against necessary libraries
target_link_libraries(copula_calculations PRIVATE
    Python3::Python # Modern CMake target for Python libs
    ${Boost_LIBRARIES} # Use the general Boost libraries variable if needed, though often implicit linking works
    boost_math_c99     # Explicitly link the C99 math library
    boost_system       # Explicitly link boost_system
)

# Add a separate executable target for try2.cpp
add_executable(optimizer_cpp try2.cpp)

# Apply compiler options to the new executable target
target_compile_options(optimizer_cpp PRIVATE -O3 -DNDEBUG)

# Link the new executable against Boost.Math, Boost.System, and OpenMP
target_link_libraries(optimizer_cpp PRIVATE
    ${Boost_LIBRARIES} # Use the general Boost libraries variable
    boost_math_c99     # Explicitly link the C99 math library
    boost_system       # Explicitly link boost_system as per original g++ command
    OpenMP::OpenMP_CXX
)

# Optional: Install the module to a specific location
# install(TARGETS copula_calculations DESTINATION your_install_path)
