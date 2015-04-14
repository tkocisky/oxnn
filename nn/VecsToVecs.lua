-- Authors: Tomas Kocisky
--
-- Efficiently maps (minibatches of) sequences of vectors to sequences of
-- vectors using a given module.
--
-- A sequence can be a table with n number of b x f tensors, or
-- a tensor of n x b x f or b x n x f, on both input and output.
--
-- Input/output type can be a 'table' or a 'tensor'.
-- Format: n - elements of a sequence, b - batch dimension, f - features.
--         Supported values are 'nbf' (necessary for table input/output),
--                              'bnf'.
-- The given module has to map m x inputSize to m x outputSize.
-- All output tensors are contiguous.
--
-- The input is always made into one 2D tensor of size -1 x inputSize.
--
local VecsToVecs, parent = torch.class('oxnn.VecsToVecs', 'nn.Container')

function VecsToVecs:__init(inputFormat, inputSize, outputType, outputFormat,
                           outputSize, module)
   parent.__init(self)

   assert(inputFormat == 'nbf' or inputFormat == 'bnf')
   self._inputFormat = inputFormat
   self._inputSize = inputSize

   self._outputType = outputType
   assert(outputFormat == 'nbf' or outputFormat == 'bnf')
   self._outputFormat = outputFormat
   self._outputSize = outputSize
   local otable = self._outputType == 'table'
   assert(not otable or self._outputFormat == 'nbf')

   self._module = module
   self.modules = { self._module }

   self._input = torch.Tensor()
   self._gradOutput = torch.Tensor()

   -- When output is a table, the output tensors are narrows of an internal
   -- tensor. In case the gradOutput is stored inplace of the output tensors,
   -- this uses the internal output tensor as the gradOutput. This avoids a copy
   -- from a table of gradOutput tensors to a joint gradOutput tensor on the
   -- backward pass.
   self.on_table_output_use_output_as_gradOutput = false
end

function VecsToVecs:updateOutput(input)
   local itable = torch.type(input) == 'table'
   local otable = self._outputType == 'table'
   local i = self._inputFormat
   local o = self._outputFormat
   assert(not itable or self._inputFormat == 'nbf')
   assert(not otable or self._outputFormat == 'nbf')
   if itable and torch.type(self.gradInput) ~= 'table' then
      self.gradInput = {}
   end

   local b, n
   if itable then
      b = input[1]:size(1)
      n = #input
   elseif i == 'bnf' then
      if input:dim() == 3 then
         b = input:size(1)
         n = input:size(2)
         assert(input:size(3) == self._inputSize)
      elseif input:dim() == 2 then
         b = input:size(1)
         n = input:size(2) / self._inputSize
      else
         error('Need input of 2 or 3 dimensions.')
      end
   elseif i == 'nbf' then
      if input:dim() == 3 then
         n = input:size(1)
         b = input:size(2)
         assert(input:size(3) == self._inputSize)
      elseif input:dim() == 2 then
         n = input:size(1)
         b = input:size(2) / self._inputSize
      else
         error('Need input of 2 or 3 dimensions.')
      end
   end
   self._b = b
   self._n = n

   if itable and o == 'nbf' then
      self._input:resize(b*n, self._inputSize)
      local input_ = self._input:view(n, b, self._inputSize)
      for i,t in ipairs(input) do
         input_:select(1, i):copy(t)
      end
   elseif itable and o == 'bnf' then
      self._input:resize(b*n, self._inputSize)
      local input_ = self._input:view(b, n, self._inputSize)
      for i,t in ipairs(input) do
         input_:select(2, i):copy(t)
      end
   elseif not itable and i == o then
      self._input:set(input:view(-1, self._inputSize))
   elseif not itable and i ~= o then
      self._input:resize(b*n, self._inputSize)
      self._input:copy(input:transpose(1,2))
      self._input:set(self._input:view(-1, self._inputSize))
   end

   local output = self._module:updateOutput(self._input)

   if otable then
      self.output = {}
      for i = 1,n do
         self.output[i] = self.output[i] or output.new()
         self.output[i]:set(output:view(n, b, self._outputSize):select(1, i))
      end
   elseif o == 'bnf' then
      self.output:set(output:view(b, n, self._outputSize))
   elseif o == 'nbf' then
      self.output:set(output:view(n, b, self._outputSize))
   end

   return self.output
end

function VecsToVecs:updateGradInput(input, gradOutput)
   local itable = torch.type(input) == 'table'
   local otable = self._outputType == 'table'
   local i = self._inputFormat
   local o = self._outputFormat
   local b = self._b
   local n = self._n

   if otable then
      if self.on_table_output_use_output_as_gradOutput then
         self._gradOutput:set(self._module.output)
      else
         self._gradOutput:resize(b*n, self._outputSize)
         local gradOutput_ = self._gradOutput:view(n, b, self._outputSize)
         for i,t in ipairs(gradOutput) do
            gradOutput_:select(1, i):copy(t)
         end
      end
   else
      self._gradOutput:set(gradOutput:view(-1, self._outputSize))
   end

   local gradInput = self._module:updateGradInput(self._input, self._gradOutput)

   if itable and o == 'nbf' then
      self.gradInput = {}
      local gradInput_ = gradInput:view(n, b, self._inputSize)
      for i = 1,#input do
         self.gradInput[i] = self.gradInput[i] or input[1].new()
         self.gradInput[i]:set(gradInput_:select(1,i))
      end
   elseif itable and o == 'bnf' then
      self.gradInput = {}
      self._gradInput_copy = self.gradInput_copy or gradInput.new()
      local gradInput_ = gradInput:view(b, n, self._inputSize):transpose(1,2)
      self._gradInput_copy:resizeAs(gradInput_):copy(gradInput_)
      for i,t in ipairs(input) do
         self.gradInput[i] = self.gradInput[i] or input[1].new()
         self.gradInput[i]:set(self._gradInput_copy:select(1,i))
      end
   elseif not itable and i == o then
      self.gradInput:set(gradInput:viewAs(input))
   elseif not itable and i ~= o then
      self.gradInput:resizeAs(input)
      self.gradInput:copy(gradInput:transpose(1,2))
      self.gradInput:set(self.gradInput:viewAs(input))
   end

   return self.gradInput
end

function VecsToVecs:accGradParameters(input, gradOutput, scale)
   self._module:accGradParameters(self._input, self._gradOutput, scale)
end

function VecsToVecs:parameters()
   return self._module:parameters()
end
