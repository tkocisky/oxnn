// modified from the oxnn package

#include "utils.h"
#include<assert.h>

static int oxnn_cuda_cudaStreamCreate(lua_State *L) {
  cudaStream_t *stream = (cudaStream_t *)lua_newuserdata(L, sizeof(cudaStream_t));
  cudaError_t err = cudaStreamCreate(stream);

  // check for errors
  if (err != cudaSuccess) {
    printf("error in cudaStreamCreate: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  return 1;
}

static int oxnn_cuda_cudaStreamDestroy(lua_State *L) {
  cudaStream_t *stream = (cudaStream_t *)lua_touserdata(L, 1);
  cudaError_t err = cudaStreamDestroy(*stream);

  // check for errors
  if (err != cudaSuccess) {
    printf("error in cudaStreamDestroy: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  return 1;
}

static int oxnn_cuda_cudaStreamSynchronize(lua_State *L) {
  cudaStream_t *stream = (cudaStream_t *)lua_touserdata(L, 1);
  cudaError_t err = cudaStreamSynchronize(*stream);

  // check for errors
  if (err != cudaSuccess) {
    printf("error in cudaStreamSynchronize: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  return 1;
}

static int oxnn_cuda_cublasGetStream(lua_State *L) {
  THCState *state = getCutorchState(L);
  THCBlasState *blas_state = state->blasState;
  cublasHandle_t *handle = blas_state->current_handle;

  cudaStream_t *stream = (cudaStream_t *)lua_newuserdata(L, sizeof(cudaStream_t));
  THCublasCheck(cublasGetStream(*handle, stream));
  return 1;
}

static int oxnn_cuda_cublasSetStream(lua_State *L) {
  THCState *state = getCutorchState(L);
  THCBlasState *blas_state = state->blasState;
  cublasHandle_t *handle = blas_state->current_handle;

  cudaStream_t *stream = (cudaStream_t *)lua_touserdata(L, 1);
  THCublasCheck(cublasSetStream(*handle, *stream));
  return 1;
}

static const struct luaL_Reg oxnn_cuda__[] = {
    {"cudaStreamCreate", oxnn_cuda_cudaStreamCreate},
    {"cudaStreamDestroy", oxnn_cuda_cudaStreamDestroy},
    {"cudaStreamSynchronize", oxnn_cuda_cudaStreamSynchronize},
    {"cublasSetStream", oxnn_cuda_cublasSetStream},
    {"cublasGetStream", oxnn_cuda_cublasGetStream},
    {NULL, NULL}};

void oxnn_cuda_init(lua_State *L) {
  lua_getglobal(L, "oxnn");
  luaL_register(L, NULL, oxnn_cuda__);
  lua_pop(L, 1);
}
