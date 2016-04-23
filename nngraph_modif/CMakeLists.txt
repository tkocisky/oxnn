FILE(GLOB luasrc *.lua)
ADD_TORCH_PACKAGE(oxnn/nngraph_modif "${src}" "${luasrc}")
