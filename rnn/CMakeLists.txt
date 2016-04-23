FILE(GLOB luasrc *.lua)
ADD_TORCH_PACKAGE(oxnn/rnn "${src}" "${luasrc}")
