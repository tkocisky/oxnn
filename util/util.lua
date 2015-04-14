
-- from nnx.Recurrent, modified with set argument
function oxnn.recursiveClone(t, set)
   set = set or function(t) return t:clone() end
   local clone
   if torch.type(t) == 'table' then
      clone = {}
      for i = 1, #t do
         clone[i] = oxnn.recursiveClone(t[i], set)
      end
   else
      if torch.typename(t) and torch.typename(t):find('torch%..+Tensor') then
         if set then clone = set(t) end
      elseif torch.type(t) == 'number' then
         if set then clone = set(t) end
      else
         error('Unexpected element.')
      end
   end
   return clone
end

function oxnn.RecursiveTensorAdd(dest, src, set_not_add_for)
   if torch.type(dest) == 'table' then
      for i,v in ipairs(dest) do
         if set_not_add_for and dest[i] == set_not_add_for then
            dest[i] = src[i]
         else
            oxnn.RecursiveTensorAdd(dest[i], src[i])
         end
      end
   else
      dest:add(src)
   end
end

function oxnn.RecursiveResizeZero(dest, idx, t)
   if torch.type(t) == 'table' then
      dest[idx] = dest[idx] or {}
      for i = 1, #t do
         oxnn.RecursiveResizeZero(dest[idx], i, t[i])
      end
   else
      if torch.typename(t) and torch.typename(t):find('torch%..+Tensor') then
         dest[idx] = dest[idx] or t.new()
         dest[idx]:resizeAs(t):zero()
      elseif torch.type(t) == 'number' then
         dest[idx] = 0
      else
         error('this shouldn\'t happen')
      end
   end
end


function oxnn.recprint(t, m)
   m = m or ''
   if torch.type(t) == 'table' then
      for i,v in ipairs(t) do oxnn.recprint(v, m..':'..i) end
   else
      print(m, t)
   end
end

function oxnn.recprinttype(t, m)
   m = m or ''
   if torch.type(t) == 'table' then
      for i,v in ipairs(t) do oxnn.recprinttype(v, m..':'..i) end
   else
      if torch.type(t):match('torch%..*Tensor') then
         print(m, torch.type(t), t:size())
      else
         print(m, torch.type(t))
      end
   end
end

function oxnn.manualSeed(seed)
   math.randomseed(seed)
   torch.manualSeed(seed)
   if package.loaded.cutorch then
      cutorch.manualSeed(seed)
      torch.zeros(1,1):cuda():uniform()
   end
end

function oxnn.OneOrTableOfClones(n, module)
   assert(n >= 1)
   if n == 1 then
      return module
   else
      local res = { module }
      for i = 2,n do
         res[#res+1] = module:clone()
      end
      return res
   end
end

function oxnn.TableOfClones(n, module)
   assert(n >= 1)
   local res = { module }
   for i = 2,n do
      res[#res+1] = module:clone()
   end
   return res
end

function oxnn.logadd(log_a, log_b)
   -- returns log(a+b)
   if log_a >= log_b then
      return log_a + math.log(1 + math.exp(log_b - log_a))
   else
      return log_b + math.log(1 + math.exp(log_a - log_b))
   end
end

function oxnn.logsum(log_as)
   local res = log_as[1]
   for i = 2, #log_as do
      res = oxnn.logadd(res, log_as[i])
   end
   return res
end

function oxnn.flatten(t)
   local res = {}
   for _,v in ipairs(t) do
      if torch.type(t) == 'table' then
         for _,w in ipairs(oxnn.flatten(v)) do
            table.insert(res, w)
         end
      else
         table.insert(res, v)
      end
   end
   return res
end
