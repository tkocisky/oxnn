-- Authors: Tomas Kocisky


local TextUtil = torch.class('oxnn.TextUtil')

-- Reads a file to list of lists of tokens, one list for each line.
function TextUtil.ReadTokenizedLines(filename)
   local result = {}

   for line in io.open(filename, 'r'):lines() do
      local tokens = {}
      for token in string.gmatch(line, '%S+') do tokens[#tokens + 1] = token end
      result[#result + 1] = tokens
   end
   return result
end

-- Reads a file to list of tokens.
function TextUtil.ReadTokenized(filename)
   local result = {}

   local file = assert(io.open(filename, 'r'))
   local contents = file:read('*a')
   file:close()

   for token in string.gmatch(contents, '%S+') do result[#result + 1] = token end
   return result
end

function TextUtil.ReadTSVLines(filename, sep)
   sep = sep or '\t'
   local result = {}

   for line in io.open(filename, 'r'):lines() do
      result[#result + 1] = string.split(line, sep)
   end
   return result
end

function TextUtil.WriteLines(file, lines)
   local f = assert(io.open(file, 'w'))
   for _,line in ipairs(lines) do
      f:write(line .. '\n')
   end
   f:close()
end

-- Given sequence of elements, splits it into sequence of sequences of elements
-- after each END element. (Keep END at the end of the sequence.)
function TextUtil.PartitionSequenceAfter(seq, ENDs)
   local ENDs = torch.type(ENDs)=='table' and ENDs or {ENDs}
   local res = {}
   local s = {}
   for i,id in ipairs(seq) do
      s[#s+1] = id
      if u.contains(ENDs, id) then
         res[#res+1] = s
         s = {}
      end
   end
   if #s > 0 then res[#res+1] = s end
   return res
end
