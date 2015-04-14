local precision = 1e-5
local t = oxnn.mytester

function oxnn.tests.oxnn_LinearBlockDiagonal()
   local p = { { 1, 1, 1, 1},
               { 1, 5, 1, 3},
               { 2, 2, 2, 1},
               { 2, 2, 2, 5},
               { 2, 6, 8, 5},
               { 3, 6, 9, 4},
               { 4, 32, 64, 10},
               { 4, 64, 32, 10},
               { 4, 64, 64, 10},
               { 8, 64, 64, 12},
             }
   for _,v in ipairs(p) do
      local b, idim, odim, batch = unpack(v)

      local C = function(a) return oxnn.cu:MaybeType(a) end

      local lb = C(oxnn.LinearBlockDiagonal(idim, odim, b))
      local l = C(nn.Linear(idim, odim))

      lb:zeroGradParameters()
      l:zeroGradParameters()

      l.bias:copy(lb.bias)
      l.weight:zero()
      for i=1,b do
         l.weight:narrow(1, (i-1)*odim/b+1, odim/b)
                 :narrow(2, (i-1)*idim/b+1, idim/b)
                 :copy(lb.weight[i])
      end

      local input = C(torch.Tensor(batch,idim):uniform(-1,1))
      lb:forward(input)
      l:forward(input)
      t:assertTensorEq(lb.output, l.output, precision)

      local g = C(torch.Tensor(batch,odim):uniform(-1,1))
      lb:backward(input, g)
      l:backward(input, g)
      t:assertTensorEq(lb.gradInput, l.gradInput, precision)
      t:assertTensorEq(lb.gradBias, l.gradBias, precision)
      for i=1,b do
         t:assertTensorEq(
            l.gradWeight:narrow(1, (i-1)*odim/b+1, odim/b)
                        :narrow(2, (i-1)*idim/b+1, idim/b),
            lb.gradWeight[i],
            precision)
      end
   end
end
