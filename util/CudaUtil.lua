local CudaUtil = torch.class('oxnn.CudaUtil')

function CudaUtil:__init(use_cuda)
   self.use_cuda = use_cuda
   self.MaybeCopyToFromCuda_CPUTensor = torch.getdefaulttensortype()
   self.MaybeCopyToFromCuda_GPUTensor = 'torch.CudaTensor'
end

function CudaUtil:TensorType()
   if self.use_cuda then
      return 'torch.CudaTensor'
   end
   return torch.getdefaulttensortype()
end

function CudaUtil:MaybeType(mlp)
   if self.use_cuda then
      local mlp_
      if type(mlp) == 'string' and mlp:find('torch%..+Tensor') then
         return 'torch.CudaTensor'
      elseif mlp.cuda then
         mlp_ = mlp:cuda()
      elseif torch.typename(mlp)
            and torch.typename(mlp):find('torch%..+Tensor') then
         mlp_ = mlp:type('torch.CudaTensor')
      end
      mlp = mlp_ or mlp
   end
   assert(mlp)
   return mlp
end

function CudaUtil:MaybeCopyToCuda(mlp)
   if self.use_cuda then
      mlp:add(nn.Copy(self.MaybeCopyToFromCuda_CPUTensor,
                      self.MaybeCopyToFromCuda_GPUTensor))
   end
   return mlp
end

function CudaUtil:MaybeCopyFromCuda(mlp)
   if self.use_cuda then
      mlp:add(nn.Copy(self.MaybeCopyToFromCuda_GPUTensor,
                      self.MaybeCopyToFromCuda_CPUTensor))
   end
   return mlp
end
