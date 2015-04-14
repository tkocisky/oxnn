#include "luaT.h"
#include "THC.h"
#include "THLogAdd.h" /* DEBUG: WTF */

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

#include "utils.c"
#include "ClassNLLCriterionD.cu"
//#include "cuda.cu"
#include "LSTM12Part2.cu"

LUA_EXTERNC DLL_EXPORT int luaopen_liboxnn_cuda(lua_State *L);

int luaopen_liboxnn_cuda(lua_State *L)
{

  lua_newtable(L);

  //oxnn_cuda_init(L);
  oxnn_ClassNLLCriterionD_init(L);
  oxnn_LSTM12Part2_init(L);

  return 1;
}
