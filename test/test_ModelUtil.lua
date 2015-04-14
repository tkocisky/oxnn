local precision = 1e-5
local t = oxnn.mytester

function oxnn.tests.oxnn_ModelUtil_LSTM12cl()
   local dim = 10

   local C = function(a) return oxnn.cu:MaybeType(a) end

   local b = C(oxnn.ModelUtil.LSTMCell1(nn.Linear(dim,dim), nn.Linear(dim,dim)))
   local bw,bdw = b:getParameters()
   local bp = b:parameters()
   bw:uniform(-1,1)

   local a = C(oxnn.ModelUtil.LSTMCell12cl(dim))
   local aw,adw = a:getParameters()
   local ap = a:parameters()

   ap1_from_bp = {5, 9, 13, 1}
   ap2_from_bp = {7, 11, 15, 3}

   for k,i in ipairs(ap1_from_bp) do
      for j = 1,dim do
         ap[1][k+(j-1)*4]:copy(bp[i][j])
         ap[2][k+(j-1)*4] = bp[i+1][j]
      end
   end
   for k,i in ipairs(ap2_from_bp) do
      for j = 1,dim do
         ap[3][k+(j-1)*4]:copy(bp[i][j])
         ap[4][k+(j-1)*4] = bp[i+1][j]
      end
   end

   local i = C(torch.Tensor(1,dim):uniform(-1,1))
   local init_c = C(torch.Tensor(1,dim):uniform(-1,1))
   local init_h = C(torch.Tensor(1,dim):uniform(-1,1))
   local input = {{init_h, init_c}, i}
   local aa = a:forward(input)
   local bb = b:forward(input)
   t:assertTensorEq(aa[1], bb[1], precision)
   t:assertTensorEq(aa[2], bb[2], precision)

   local grad_c = C(torch.Tensor(1,dim):uniform(-1,1))
   local grad_h = C(torch.Tensor(1,dim):uniform(-1,1))
   local gaa = a:backward(input, {grad_h,grad_c})
   local gbb = b:backward(input, {grad_h,grad_c})
   t:assertTensorEq(gaa[1][1], gbb[1][1], precision)
   t:assertTensorEq(gaa[1][2], gbb[1][2], precision)
   t:assertTensorEq(gaa[2], gbb[2], precision)
end

function oxnn.tests.oxnn_ModelUtil_LSTM12c()
   local dim = 10

   local C = function(a) return oxnn.cu:MaybeType(a) end

   if oxnn.cu.use_cuda then
      local a = oxnn.LSTM12Part2(dim)
      local b = C(oxnn.LSTM12Part2(dim))

      local prev_c = torch.Tensor(1,dim):uniform(-1,1)
      local raw_gates = torch.Tensor(1,4*dim):uniform(-1,1)

      local inputa = { prev_c:clone(), raw_gates:clone() }
      local inputb = { prev_c:clone():cuda(), raw_gates:clone():cuda() }

      local aa = a:forward(inputa)
      local bb = b:forward(inputb)
      t:assertTensorEq(aa[1]:float(), bb[1]:float(), precision)
      t:assertTensorEq(aa[2]:float(), bb[2]:float(), precision)

      local grad_c = torch.Tensor(1,dim):uniform(-1,1)
      local grad_h = torch.Tensor(1,dim):uniform(-1,1)

      local gaa = a:backward(inputa, { grad_h:clone(), grad_c:clone() })
      local gbb = b:backward(inputb, { grad_h:clone():cuda(),
                                       grad_c:clone():cuda() })
      t:assertTensorEq(gaa[1]:float(), gbb[1]:float(), precision)
      t:assertTensorEq(gaa[2]:float(), gbb[2]:float(), precision)
   end
end
