# Generate the sphinx config file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in ${CMAKE_CURRENT_BINARY_DIR}/conf.py @ONLY)

# ---------------------------------
# Top Sphinx target
# ---------------------------------
file(GLOB_RECURSE sources *.rst)

set(sphinx_top ${CMAKE_CURRENT_BINARY_DIR}/html/contents.html)
add_custom_command(OUTPUT ${sphinx_top} DEPENDS ${sources}
                   COMMAND ${TRIQS_SPHINXBUILD_EXECUTABLE} -c . -j8 -b html ${CMAKE_CURRENT_BINARY_DIR} html)
add_custom_target(docs_sphinx ALL DEPENDS ${sphinx_top} ${CMAKE_CURRENT_BINARY_DIR})

# ------------------------------------------------------------------------------------------------
# Copy the directory substructure and link all files with relevant extensions
# cp_rs is a script in cpp2py/bin, it mimics cp -rs on Linux
# and filters the relevant extension
# ------------------------------------------------------------------------------------------------
set(EXT_FOR_DOC "rst png txt css_t conf css js gif jpg py html bib md")
execute_process(COMMAND cp_rs ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${EXT_FOR_DOC})

# ---------------------------------
# Install
# ---------------------------------
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html/ COMPONENT documentation DESTINATION share/doc/w2dynamics_interface
  FILES_MATCHING
  REGEX "\\.(html|pdf|png|gif|jpg|js|xsl|css|py|txt|inv|bib)$"
  PATTERN "_*"
)
