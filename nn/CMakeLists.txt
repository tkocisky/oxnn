FILE(GLOB luasrc *.lua)
ADD_TORCH_PACKAGE(oxnn/nn "${src}" "${luasrc}")
