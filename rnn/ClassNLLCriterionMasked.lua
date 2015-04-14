-- Authors: Tomas Kocisky
--
-- Give a mask (0/1) exludes the NLL loss for some of the examples in the
-- minibatch. Returns a tensor of losses for each example in the minibatch.
--
local ClassNLLCriterionMasked, parent = torch.class('oxnn.ClassNLLCriterionMasked', 'nn.ClassNLLCriterion')

function ClassNLLCriterionMasked:__init()
   parent.__init(self)
   self.sizeAverage = false
   self.outputTensor = torch.Tensor(1)
   self.alpha = -1
   self.vectorOutput = false or (params and params.batchscore)
end

function ClassNLLCriterionMasked:updateOutput(input, target)
   local mask = target[2]
   assert(mask:dim() == 1)
   target = target[1]
   if input:dim() == 2 then
      if input:type() == 'torch.CudaTensor' and not self.vectorOutput then
         if torch.type(target) ~= 'torch.CudaTensor' then
            target = target:float():cuda()
         end
         if not self._target_masked then self._target_masked = input.new() end
         if target:isContiguous() then
            self._target_masked:cmul(target, mask)
         else
            self._target_masked:resizeAs(target):copy(target):cmul(mask)
         end
         self.alpha = 1 / mask:sum()
         -- the following function multiplies the double precision result by
         -- self.alha (if alpha > 0) before converting to single precition.
         input.oxnn.ClassNLLCriterionD_updateOutput(self, input,
                                                    self._target_masked)
         self.output = self.outputTensor[1]
         if self.alpha > 0 then
            self.output = self.output / self.alpha
         end

      else
         local targeti = target:clone():view(-1):float():long()
         for i = 1,targeti:size(1) do
            targeti[i] = targeti[i] + (i-1) * input:size(2)
         end
         local output = input:view(-1):index(1, targeti):cmul(mask):mul(-1)
         if self.sizeAverage then
            output = output / target:size(1)
         end
         if not self.vectorOutput then
            self.output = output:sum()
         else
            self.output = output
         end
      end
   else
      error('matrix expected')
   end
   return self.output
end

function ClassNLLCriterionMasked:updateGradInput(input, target)
   self.gradInput:resizeAs(input)
   self.gradInput:zero()

   local mask = target[2]
   assert(mask:dim() == 1)
   target = target[1]

   if input:type() == 'torch.CudaTensor' then
      input.oxnn.ClassNLLCriterionD_updateGradInput(self, input,
                                                    self._target_masked)
      return self.gradInput
   else
      local z = -1
      if self.sizeAverage then
         z = z / target:size(1)
      end
      for i=1,target:size(1) do
         self.gradInput[i][target[i]] = z * mask[i]
      end
      return self.gradInput
   end
end
