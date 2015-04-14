-- Modified updateOutput of nn.Linear so that it doesn't zero it's self.output
-- tensor before adding the result into it. Useful for oxnn.LinearCAddInplace.
--
-- The removed code is comented out.
--
local LinearNoOutputZero, parent = torch.class('oxnn.LinearNoOutputZero', 'nn.Linear')

function LinearNoOutputZero:__init(inputSize, outputSize)
   parent.__init(self, inputSize, outputSize)
end


function LinearNoOutputZero:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.bias:size(1))
      self.output:add--[[copy]](self.bias)
      self.output:addmv(1, self.weight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nunit = self.bias:size(1)
      self.output:resize(nframe, nunit)
      if not self.addBuffer or self.addBuffer:size(1) ~= nframe then
         self.addBuffer = input.new(nframe):fill(1)
      end
      if nunit == 1 then
         -- Special case to fix output size of 1 bug:
         self.output:copy(self.bias:view(1,nunit):expand(#self.output))
         self.output:select(2,1):addmv(1, input, self.weight:select(1,1))
      else
         self.output--[[:zero()]]:addr(1, self.addBuffer, self.bias)
         self.output:addmm(1, input, self.weight:t())
      end
   else
      error('input must be vector or matrix')
   end

   return self.output
end
