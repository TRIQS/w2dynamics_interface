# Generate the sphinx config file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in ${CMAKE_CURRENT_BINARY_DIR}/conf.py @ONLY)

# ---------------------------------
# Top Sphinx target
# ---------------------------------
# Sphinx has internal caching, always run it
add_custom_target(docs_sphinx ALL)
add_custom_command(
  TARGET docs_sphinx
  COMMAND PYTHONPATH=${PROJECT_BINARY_DIR}/python:${PROJECT_BINARY_DIR}/w2dyn_project/src:$ENV{PYTHONPATH} ${TRIQS_SPHINXBUILD_EXECUTABLE} -c . -j8 -b html ${CMAKE_CURRENT_SOURCE_DIR} html
)

# ---------------------------------
# Install
# ---------------------------------
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html/ COMPONENT documentation DESTINATION share/doc/w2dynamics_interface
  FILES_MATCHING
  REGEX "\\.(html|pdf|png|gif|jpg|svg|js|xsl|css|py|txt|inv|bib)$"
  PATTERN "_*"
)
