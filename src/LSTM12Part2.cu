/*
 * Authors: Tomas Kocisky
 *
 * Faster component-wise "part 2" of LSTM.
 */
#include "utils.h"

template<class F>
struct lstmpart2Output_functor
{
  inline __host__ __device__ F s(const F& f) const {
    return 1./(1. + exp(-f));
  }
  inline __host__ __device__ F ds(const F& sf) const {
    return (1. - sf) * sf;
  }
  inline __host__ __device__ F t(const F& f) const {
    return tanh(f);
  }
  inline __host__ __device__ F dt(const F& tf) const {
    return 1. - tf * tf;
  }

  __host__ __device__ F operator()(const F& pc, const F& fg, const F& ig,
                                   const F& i, const F& og, F& nc, F& nh) const
  {
    nc = s(fg) * pc + s(ig) * t(i);
    nh = s(og) * t(nc);
    return 0;
  }
};

template<class F>
struct lstmpart2GradInput_functor
{
  inline __host__ __device__ F s(const F& f) const {
    return 1./(1. + exp(-f));
  }
  inline __host__ __device__ F ds(const F& sf) const {
    return (1. - sf) * sf;
  }
  inline __host__ __device__ F t(const F& f) const {
    return tanh(f);
  }
  inline __host__ __device__ F dt(const F& tf) const {
    return 1. - tf * tf;
  }

  __host__ __device__ F operator()(
      F& pc, F& fg, F& ig, F& i, F& og,
      const F& gonc, const F& gonh
      ) const
  {
    float sfg = s(fg);
    float sig = s(ig);
    float sog = s(og);
    float ti = t(i);
    float tnc = t(sfg * pc + sig * ti);
    og = gonh * tnc * ds(sog);
    float gnc = gonc + gonh * sog * dt(tnc);
    fg = gnc * pc * ds(sfg);
    pc = gnc * sfg;
    ig = gnc * ti * ds(sig);
    i = gnc * sig * dt(ti);
    return 0;
  }
};

// from https://thrust.github.io/doc/classthrust_1_1iterator__adaptor.html
#include <thrust/iterator/iterator_adaptor.h>
template<typename Iterator>
class skip_iterator
  : public thrust::iterator_adaptor< skip_iterator<Iterator>, Iterator >
{
 public:
  typedef thrust::iterator_adaptor<skip_iterator<Iterator>, Iterator > super_t;

  __host__ __device__
  skip_iterator(const Iterator &x, int n) : super_t(x), begin(x), n(n) {}
  friend class thrust::iterator_core_access;
 private:
  unsigned int n;
  const Iterator begin;

  __host__ __device__
  typename super_t::reference dereference() const {
    return *(begin + (this->base() - begin) * n);
  }
};

template<typename Iterator>
skip_iterator<Iterator> make_skip_iterator(Iterator it, int n)
{ return skip_iterator<Iterator>(it,n); }

template<typename SevenaryFunction>
struct tuple7_transform_functor
{
  SevenaryFunction f;

  __host__ __device__ tuple7_transform_functor(SevenaryFunction f) : f(f) {}

  template<typename Tuple>
  inline __host__ __device__ void operator()(Tuple t) {
    f(
        thrust::get<0>(t),
        thrust::get<1>(t),
        thrust::get<2>(t),
        thrust::get<3>(t),
        thrust::get<4>(t),
        thrust::get<5>(t),
        thrust::get<6>(t)
      );
  }
};

static int oxnn_LSTM12Part2_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *prev_c = (THCudaTensor*)luaT_checkudata(L, 1,
                                                        "torch.CudaTensor");
  THCudaTensor *raw_gates = (THCudaTensor*)luaT_checkudata(L, 2,
                                                        "torch.CudaTensor");
  THCudaTensor *next_c = (THCudaTensor*)luaT_checkudata(L, 3,
                                                        "torch.CudaTensor");
  THCudaTensor *next_h = (THCudaTensor*)luaT_checkudata(L, 4,
                                                        "torch.CudaTensor");

  long size = THCudaTensor_nElement(state, prev_c);
  long size_g = THCudaTensor_nElement(state, raw_gates);
  long size_nc = THCudaTensor_nElement(state, next_c);
  long size_nh = THCudaTensor_nElement(state, next_h);

  assert(THCudaTensor_isContiguous(state, prev_c));
  assert(THCudaTensor_isContiguous(state, raw_gates));
  assert(THCudaTensor_isContiguous(state, next_c));
  assert(THCudaTensor_isContiguous(state, next_h));
  assert(size * 4 == size_g);
  assert(size == size_nc);
  assert(size == size_nh);

  typedef tuple7_transform_functor<lstmpart2Output_functor<float> >
    SevenaryTransformFunctor;

  thrust::device_ptr<float> prev_c_data(THCudaTensor_data(state, prev_c));
  thrust::device_ptr<float> raw_gates_data(THCudaTensor_data(state, raw_gates));
  thrust::device_ptr<float> next_c_data(THCudaTensor_data(state, next_c));
  thrust::device_ptr<float> next_h_data(THCudaTensor_data(state, next_h));
  thrust::for_each(
                   thrust::make_zip_iterator(thrust::make_tuple(
                       prev_c_data,
                       make_skip_iterator(raw_gates_data+0, 4),
                       make_skip_iterator(raw_gates_data+1, 4),
                       make_skip_iterator(raw_gates_data+2, 4),
                       make_skip_iterator(raw_gates_data+3, 4),
                       next_c_data,
                       next_h_data
                       )),
                   thrust::make_zip_iterator(thrust::make_tuple(
                       prev_c_data+size,
                       make_skip_iterator(raw_gates_data+0, 4),
                       make_skip_iterator(raw_gates_data+1, 4),
                       make_skip_iterator(raw_gates_data+2, 4),
                       make_skip_iterator(raw_gates_data+3, 4),
                       next_c_data,
                       next_h_data
                       )),
                   SevenaryTransformFunctor(lstmpart2Output_functor<float>())
                  );
  return 1;
}


static int oxnn_LSTM12Part2_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *prev_c = (THCudaTensor*)luaT_checkudata(L, 1,
                                                        "torch.CudaTensor");
  THCudaTensor *raw_gates = (THCudaTensor*)luaT_checkudata(L, 2,
                                                        "torch.CudaTensor");
  THCudaTensor *GOnext_c = (THCudaTensor*)luaT_checkudata(L, 3,
                                                        "torch.CudaTensor");
  THCudaTensor *GOnext_h = (THCudaTensor*)luaT_checkudata(L, 4,
                                                        "torch.CudaTensor");

  long size = THCudaTensor_nElement(state, prev_c);
  long size_g = THCudaTensor_nElement(state, raw_gates);
  long size_nc = THCudaTensor_nElement(state, GOnext_c);
  long size_nh = THCudaTensor_nElement(state, GOnext_h);

  assert(THCudaTensor_isContiguous(state, prev_c));
  assert(THCudaTensor_isContiguous(state, raw_gates));
  assert(THCudaTensor_isContiguous(state, GOnext_c));
  assert(THCudaTensor_isContiguous(state, GOnext_h));
  assert(size * 4 == size_g);
  assert(size == size_nc);
  assert(size == size_nh);


  typedef tuple7_transform_functor<lstmpart2GradInput_functor<float> >
    SevenaryTransformFunctor;

  thrust::device_ptr<float> prev_c_data(THCudaTensor_data(state, prev_c));
  thrust::device_ptr<float> raw_gates_data(THCudaTensor_data(state, raw_gates));
  thrust::device_ptr<float> GOnext_c_data(THCudaTensor_data(state, GOnext_c));
  thrust::device_ptr<float> GOnext_h_data(THCudaTensor_data(state, GOnext_h));
  thrust::for_each(
                   thrust::make_zip_iterator(thrust::make_tuple(
                       prev_c_data,
                       make_skip_iterator(raw_gates_data+0, 4),
                       make_skip_iterator(raw_gates_data+1, 4),
                       make_skip_iterator(raw_gates_data+2, 4),
                       make_skip_iterator(raw_gates_data+3, 4),
                       GOnext_c_data,
                       GOnext_h_data
                       )),
                   thrust::make_zip_iterator(thrust::make_tuple(
                       prev_c_data+size,
                       make_skip_iterator(raw_gates_data+0, 4),
                       make_skip_iterator(raw_gates_data+1, 4),
                       make_skip_iterator(raw_gates_data+2, 4),
                       make_skip_iterator(raw_gates_data+3, 4),
                       GOnext_c_data,
                       GOnext_h_data
                       )),
                   SevenaryTransformFunctor(
                     lstmpart2GradInput_functor<float>())
                  );
  return 1;
}

static const struct luaL_Reg oxnn_LSTM12Part2__ [] = {
  {"LSTM12Part2_updateOutput", oxnn_LSTM12Part2_updateOutput},
  {"LSTM12Part2_updateGradInput", oxnn_LSTM12Part2_updateGradInput},
  {NULL, NULL}
};

static void oxnn_LSTM12Part2_init(lua_State *L) {
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, oxnn_LSTM12Part2__, "oxnn");
  lua_pop(L,1);
}
