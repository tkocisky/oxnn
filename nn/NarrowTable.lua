-- Authors: Tomas Kocisky
--
-- Multiple narrows of a tensor.
--
-- This is more memory efficient than separate nn.Narrow s.
--
local NarrowTable, parent = torch.class('oxnn.NarrowTable', 'nn.Module')

function NarrowTable:__init(dimension, slices, copy_output, zero_gradInput)
   parent.__init(self)
   self.dimension = dimension
   self.slices = slices
   self.copy_output = copy_output or false
   self.zero_gradInput = zero_gradInput == nil and true or zero_gradInput
   self.output = {}
end

function NarrowTable:updateOutput(input)
   self.output = {}
   local dimension = self.dimension
   local out = self.output
   for i = 1, #self.slices do
      local start, length = unpack(self.slices[i])

      if not out[i] then out[i] = input.new() end

      local t = input:narrow(dimension, start, length)
      if self.copy_output then
         out[i]:resizeAs(t):copy(t)
      else
         out[i]:set(t)
      end
   end
   return self.output
end

function NarrowTable:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   if self.zero_gradInput then
      self.gradInput:zero()
   end

   local dimension = self.dimension
   for i = 1,#self.slices do
      local start, length = unpack(self.slices[i])
      if self.zero_gradInput then
         self.gradInput:narrow(dimension, start, length):add(gradOutput[i])
      else
         self.gradInput:narrow(dimension, start, length):copy(gradOutput[i])
      end
   end
   return self.gradInput
end
