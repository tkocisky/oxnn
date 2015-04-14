-- Authors: Tomas Kocisky
--
-- Componentwise add which stores the output inplace of the first element.
--
local CAddTableNoCopyInplace, parent = torch.class('oxnn.CAddTableNoCopyInplace', 'oxnn.CAddTableNoCopy')

function CAddTableNoCopyInplace:updateOutput(input)
   self.output = input[1]
   for i=2,#input do
      self.output:add(input[i])
   end
   return self.output
end
