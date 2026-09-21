# Embeds a text file (an OpenCL kernel) into a C++ header as a raw string
# literal, so that libprimesieve carries its GPU kernels inside the binary
# and needs no data files at runtime.
#
#   cmake -DINPUT=<file> -DOUTPUT=<header> -DSYMBOL=<name> -P embed_file.cmake

if(NOT DEFINED INPUT OR NOT DEFINED OUTPUT OR NOT DEFINED SYMBOL)
  message(FATAL_ERROR "embed_file.cmake needs INPUT, OUTPUT and SYMBOL")
endif()

file(READ "${INPUT}" _content)

# The delimiter must not occur in the kernel source. Use string(FIND)
# rather than a regex so that the parenthesis needs no escaping.
string(FIND "${_content}" ")PSKERNEL" _clash)
if(NOT _clash EQUAL -1)
  message(FATAL_ERROR "${INPUT} contains the raw string delimiter )PSKERNEL")
endif()

get_filename_component(_name "${INPUT}" NAME)

file(WRITE "${OUTPUT}"
"// GENERATED from ${_name} by tools/embed_file.cmake -- do not edit.
#ifndef PRIMESIEVE_GPU_EMBEDDED_${SYMBOL}
#define PRIMESIEVE_GPU_EMBEDDED_${SYMBOL}

static const char ${SYMBOL}[] = R\"PSKERNEL(
${_content}
)PSKERNEL\";

#endif
")
