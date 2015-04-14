oxnnmc = require 'oxnn.util.models_common'
local cmd = oxnnmc.cmd

cmd:option('-worddim', 10, 'Word representation dimention.')
cmd:option('-fast', false, 'Turns of some assertions/checks.')
cmd:option('-train', false, '')
cmd:option('-stdv', 0, 'Value to pass to the nn.Module.reset')
cmd:option('-unif', 0, 'If nonzero then initializes all weights uniformly in (-unif,unif).')
cmd:option('-bsize', 1.2, 'Bucket maximal longest/shortest ratio.')
cmd:option('-maxlen', 0, 'Limit sequence length (for development).')
oxnn.Optimizer.AddCmdOptions(cmd)

oxnnmc.Init()
local loadmodel = oxnn.Optimizer.LoadModel()
oxnnmc.MaybeLog('toy1')


-- Read file using methods in TextUtil. Static here.

local d_train = {
{'lorem', 'ipsum', 'dolor', 'sit', 'amet', ',', 'consectetur', 'adipiscing', 'elit', '.',},
{'mauris', 'a', 'arcu', 'non', 'libero', 'ultrices', 'volutpat', 'vitae', 'eu', 'est', '.',},
{'etiam', 'lacinia', 'justo', 'accumsan', 'sapien', 'tincidunt', ',', 'id', 'ornare', 'dui', 'dictum', '.',},
{'morbi', 'congue', 'lacus', 'eu', 'ipsum', 'interdum', 'varius', '.',},
{'in', 'sit', 'amet', 'ex', 'quis', 'metus', 'consequat', 'rhoncus', 'volutpat', 'nec', 'elit', '.',},
{'integer', 'eleifend', 'elit', 'et', 'velit', 'tincidunt', 'iaculis', '.',},
{'mauris', 'at', 'urna', 'maximus', ',', 'tincidunt', 'libero', 'in', ',', 'sollicitudin', 'lorem', '.',},
{'aenean', 'sit', 'amet', 'libero', 'id', 'sem', 'maximus', 'placerat', '.',},
{'suspendisse', 'in', 'dui', 'et', 'ipsum', 'gravida', 'molestie', '.',},
{'in', 'vitae', 'dui', 'in', 'mi', 'porta', 'tempor', '.',},
{'cras', 'quis', 'lacus', 'laoreet', ',', 'semper', 'mauris', 'non', ',', 'euismod', 'lacus', '.',},
{'praesent', 'et', 'nisi', 'at', 'justo', 'convallis', 'imperdiet', 'non', 'non', 'ante', '.',},
{'duis', 'ornare', 'eros', 'at', 'justo', 'aliquam', 'congue', '.',},
{'nullam', 'tristique', 'felis', 'quis', 'molestie', 'porttitor', '.',},
}
local d_test = {
{'etiam', 'sit', 'amet', 'leo', 'tristique', ',', 'pulvinar', 'odio', 'quis', ',', 'imperdiet', 'metus', '.',},
{'nam', 'et', 'quam', 'vehicula', 'quam', 'imperdiet', 'pharetra', '.',},
{'ut', 'placerat', 'leo', 'id', 'magna', 'rhoncus', 'accumsan', 'a', 'ac', 'sapien', '.',},
{'pellentesque', 'id', 'nibh', 'ut', 'mauris', 'viverra', 'semper', '.',},
{'aenean', 'id', 'mi', 'at', 'erat', 'tempor', 'ornare', 'et', 'nec', 'tortor', '.',},
{'aenean', 'tristique', 'justo', 'quis', 'vestibulum', 'auctor', '.',},
}

local voc = oxnn.Vocabulary()
local UNK_ID = voc:Insert('<_UNK_>')
local DOC_SID = voc:Insert('<_DOC_START_>')
local DOC_EID = voc:Insert('<_DOC_END_>')
local PAD_ID = voc:Insert('<_PAD_>')

-- Load train data
local train = {}
for i,v in ipairs(d_train) do
   train[i] = torch.LongTensor(oxnn.Vocabulary.WordsToIds(
                 v, voc, oxnn.Vocabulary.unk_callback_insert))
end
-- Load test data
local test = {}
for i,v in ipairs(d_test) do
   test[i] = torch.LongTensor(oxnn.Vocabulary.WordsToIds(
                v, voc, function() return UNK_ID end))
end
local test_sets = {test1=test}


---- model definition ----

local mlp = nn.Sequential()

mlp:add(oxnn.SequenceOfWords{
   lookuptable = nn.Sequential():add(nn.LookupTable(voc:size(), params.worddim))
                                :add(nn.SplitTable(2)),
   recurrent = oxnn.ModelUtil.LSTMCell1(
                  nn.Linear(params.worddim, params.worddim),
                  nn.Linear(params.worddim, params.worddim),
                  true),
   output = nn.Sequential()
               :add(nn.Linear(params.worddim, voc:size()))
               :add(oxnn.LogSoftMaxInplace(true,true)),
   loss = 'nllloss',
})

mlp:add(nn.SelectTable(2))

mlp = oxnn.cu:MaybeType(mlp)

---- end model definition ----

local w,dw = mlp:getParameters()
if loadmodel then
   w:copy(loadmodel[2])
else
   mlp:reset(params.stdv > 0 and params.stdv or nil)
   if params.unif > 0 then
      w:uniform(-params.unif, params.unif)
   end
end

---- create minibatches and input ----

local function CreateInput(ba_it, batch_size, max_doc_len)
   local res
   local cu = oxnn.cu
   res =
         {[1] = {
            -- i1 (initial recurrent state)
            {{cu:MaybeType(torch.zeros(batch_size, params.worddim)),
               cu:MaybeType(torch.zeros(batch_size, params.worddim))}},
            -- i2 (docs)
            torch.LongTensor(batch_size, max_doc_len):fill(PAD_ID),
            -- i3 (lengths)
            {},
            batch_size = batch_size,  -- used in the Optimizer.optimize
            perp_output_tokens = 0,  -- for loss averaging
            },
          [2] = nil
         }
   for idx, doc_ids in ba_it do
      local doc_len = math.min(doc_ids:nElement(), max_doc_len)
      local t_doc = res[1][2]
      t_doc:select(1,idx):narrow(1,1, doc_len)
           :copy(doc_ids:narrow(1,1,doc_len))
      table.insert(res[1][3], doc_len)

      res[1].perp_output_tokens = res[1].perp_output_tokens + doc_len - 1
   end
   res[1][2] = cu:MaybeType(res[1][2])
   return res
end

local function GetDataset(data, max_batch)

   max_batch = max_batch or 1
   local batches = {}
   local b = {}
   for i=1,#data do
      table.insert(b, i)
      if #b == max_batch or i == #data then
         table.insert(batches, b)
         b = {}
      end
   end


   -- create a dataset object
   --
   -- the format is the same as for the nn.StochasticGradient
   -- https://github.com/torch/nn/blob/master/doc/training.md
   local function get_datum(t, i)
      local doc_ids = batches[i]
      local batch_size = #doc_ids

      local max_doc_len = 0
      for i = 1, batch_size do
         max_doc_len = math.max(max_doc_len, data[doc_ids[i]]:nElement())
      end
      if params.maxlen > 0 then
         max_doc_len = math.min(params.maxlen, max_doc_len)
      end

      -- iterator over elements in the minibatch
      local function ba_it(_s, idx)
         idx = idx or 0
         if idx == batch_size then return nil end
         idx = idx + 1
         return idx, data[doc_ids[idx]]
      end

      return CreateInput(ba_it, batch_size, max_doc_len)
   end

   local dataset = {}
   function dataset:size() return #batches end
   return setmetatable(dataset, {__index=get_datum})
end


---- train ----
if params.train then
   local criterion = nil
   oxnn.Optimizer.Optimize(mlp, w, dw, criterion, GetDataset, train, test_sets,
                           loadmodel,
                           oxnn.Optimizer.SimplePerplexityUpdateError,
                           oxnn.Optimizer.SimplePerplexityGetError,
                           oxnn.Optimizer.PrintPerplexityStats)
end
