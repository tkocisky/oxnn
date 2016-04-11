FILE(GLOB luasrc *.lua)
ADD_TORCH_PACKAGE(oxnn/nn_modif "${src}" "${luasrc}")
