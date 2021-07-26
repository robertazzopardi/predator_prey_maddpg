# Look for the header file.
find_path(ROBOSIM_INCLUDE_DIR NAMES robosim.h)

# Look for the library.
find_library(ROBOSIM_LIBRARY NAMES robosim)

# Handle the QUIETLY and REQUIRED arguments and set ROBOSIM_FOUND to TRUE if all
# listed variables are TRUE.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ROBOSIM DEFAULT_MSG ROBOSIM_LIBRARY
                                  ROBOSIM_INCLUDE_DIR)

# Copy the results to the output variables.
if(ROBOSIM_FOUND)
  set(ROBOSIM_LIBRARIES ${ROBOSIM_LIBRARY})
  set(ROBOSIM_INCLUDE_DIRS ${ROBOSIM_INCLUDE_DIR})
  add_library(ROBOSIM::Core UNKNOWN IMPORTED)
  set_target_properties(
    ROBOSIM::Core
    PROPERTIES IMPORTED_LOCATION "${ROBOSIM_LIBRARY}"
               INTERFACE_INCLUDE_DIRECTORIES "${ROBOSIM_INCLUDE_DIR}")
else(ROBOSIM_FOUND)
  set(ROBOSIM_LIBRARIES)
  set(ROBOSIM_INCLUDE_DIRS)
endif(ROBOSIM_FOUND)

mark_as_advanced(ROBOSIM_INCLUDE_DIRS ROBOSIM_LIBRARIES)
