-- Authors: Tomas Kocisky
--
-- Takes as input a table of losses (numbers or 1 element tensors) and returns
-- their sum as a number.
--
local SumLosses, parent = torch.class('oxnn.SumLosses', 'nn.Criterion')

function SumLosses:__init(average, lengths)
   parent.__init(self)
   self.average = average or false
   self.lengths = lengths
end

function SumLosses:updateOutput(input)
   if torch.type(input[1]) == 'number' then
      self.output = 0
   else
      if not torch.type(self.output):find('torch%..+Tensor') then
         self.output = input[1].new()
      end
      self.output:resizeAs(input[1]):zero()
   end
   for _,v in ipairs(input) do
      if torch.type(self.output) == 'number' then
         self.output = self.output + v
      else
         self.output:add(v)
      end
   end
   if self.average then
      assert(self.lengths)
      local tokens = 0
      for _,v in ipairs(self.lengths) do tokens = tokens + math.max(0, v - 1) end
      self.output = self.output / tokens
   end
   return self.output
end

function SumLosses:updateGradInput(input, target)
   assert(type(input) == 'table')
   self.gradInput = {}
   for i,v in ipairs(input) do
      if torch.typename(v) and torch.typename(v):find('torch%..+Tensor') then
         self.gradInput[i] = self.gradInput[i] or v.new()
         self.gradInput[i]:resizeAs(v):zero()
      else
         self.gradInput[i] = 0
      end
   end
   return self.gradInput
end

function SumLosses:type(type)
   self.gradInput = self.gradInput:type(type)
   return self
end
