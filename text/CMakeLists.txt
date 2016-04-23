FILE(GLOB luasrc *.lua)
ADD_TORCH_PACKAGE(oxnn/text "${src}" "${luasrc}")
