-- Authors: Tomas Kocisky
--
-- Sums a table of tensors (possible table of tables of...) and uses only the
-- examples of the minibatch as indicated by the tensor of masks 'ended'.
--
local MaskedAdd, parent = torch.class('oxnn.MaskedAdd', 'nn.Module')

function MaskedAdd:__init(ended, endedat)
   parent.__init(self)
   self.ended = ended
   self.endedat = endedat
end

function MaskedAdd:updateOutput(input)
   for i = 1,#input do
      if i == 1 then
         local masked = oxnn.recursiveClone(
            input[i], function(t)
                         return oxnn.Mask.mask(t, self.ended:select(1,i))
                      end)
         self.output = masked
      else
         oxnn.Mask.add_mask(self.output, input[i], self.ended:select(1,i))
      end
   end
   return self.output
end

function MaskedAdd:updateGradInput(input, gradOutput)
   self.gradInput = {}
   for i = 1,#input do
      self.gradInput[i] = oxnn.recursiveClone(
         gradOutput, function(t)
                        return oxnn.Mask.mask(t, self.ended:select(1,i))
                     end)
   end
   return self.gradInput
end

function MaskedAdd:type(type)
   self.output = self.output:type(type)
   self.gradInput = self.gradInput:type(type)
   return self
end
