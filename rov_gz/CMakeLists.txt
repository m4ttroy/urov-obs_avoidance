cmake_minimum_required(VERSION 3.12.2)
project(rov_gz)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find ROS 2 build packages
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_auto REQUIRED)

# Automatically find package build dependencies
ament_auto_find_build_dependencies()

# Update GZ_SIM_RESOURCE_PATH via DSV environment hooks (Must be before ament_package)
ament_environment_hooks(hooks/setup.dsv)

# ArduSub SITL build and configuration via waf
set(ARDUSUPRI_DIR "$ENV{HOME}/Ardusupri")

option(BUILD_ARDUSUB_SITL "Build ArduSub SITL with waf" OFF)
if(BUILD_ARDUSUB_SITL)
  execute_process(
    COMMAND python3 -c "import em"
    RESULT_VARIABLE EM_RESULT
    OUTPUT_QUIET ERROR_QUIET)
  if(NOT EM_RESULT EQUAL 0)
    message(FATAL_ERROR "Missing Python module 'empy'. Run ./setup.sh first.")
  endif()

  if(NOT EXISTS "${ARDUSUPRI_DIR}/build/sitl/c4che/_cache.py")
    execute_process(
      COMMAND python3 ./waf configure --board sitl
      WORKING_DIRECTORY "${ARDUSUPRI_DIR}"
      RESULT_VARIABLE WAF_CONFIGURE_RESULT)
    if(NOT WAF_CONFIGURE_RESULT EQUAL 0)
      message(FATAL_ERROR "waf configure failed in ${ARDUSUPRI_DIR}")
    endif()
  endif()

  add_custom_target(build_ardusub ALL
    COMMAND python3 ./waf sub
    WORKING_DIRECTORY "${ARDUSUPRI_DIR}"
    USES_TERMINAL)
endif()

# Installation rules for all resource directories, configs, and parameters
install(
  DIRECTORY models worlds launch config params
  DESTINATION share/${PROJECT_NAME}
  PATTERN "*~" EXCLUDE
  PATTERN "*.in" EXCLUDE)

# Linter and testing configuration
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()