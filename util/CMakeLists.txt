FILE(GLOB luasrc *.lua)
ADD_TORCH_PACKAGE(oxnn/util "${src}" "${luasrc}")
