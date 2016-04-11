-- Example creating a bidirectional LSTM encoder
require 'oxnn'

local vocabSize = 10
local hiddenSize = 5
local fwd_model = nn.Sequential()
   :add(oxnn.SequenceOfWords{
           lookuptable = nn.Sequential():add(nn.LookupTable(vocabSize, 
                                                            hiddenSize))
              :add(nn.SplitTable(2)),
           recurrent = oxnn.ModelUtil.LSTMCell1(
              nn.Linear(hiddenSize, hiddenSize),
              nn.Linear(hiddenSize, hiddenSize),
              false),
           loss = 'noloss'})
   :add(nn.SelectTable(1))
   :add(nn.SelectTable(2))

local bwd_model = nn.Sequential()
   :add(oxnn.SequenceOfWords{
           lookuptable = nn.Sequential():add(nn.LookupTable(vocabSize, 
                                                            hiddenSize))
              :add(nn.SplitTable(2)),
           recurrent = oxnn.ModelUtil.LSTMCell1(
              nn.Linear(hiddenSize, hiddenSize),
              nn.Linear(hiddenSize, hiddenSize),
              false),
           loss = 'noloss'})
   :add(nn.SelectTable(1))
   :add(nn.SelectTable(2))
                             
local input = { 
   {
      { { torch.zeros(2,hiddenSize), torch.zeros(2,hiddenSize) } },
      torch.Tensor{ 
         { 1, 7, 9, 2 },   -- sentence 1
         { 1, 3, 2, 2 }    -- sentence 2
      },
      { 4, 3 }  -- sentence lengths
   },
   {
      { { torch.zeros(2,hiddenSize), torch.zeros(2,hiddenSize) } },
      torch.Tensor{ 
         { 1, 9, 7, 2 },   -- sentence 1
         { 1, 3, 2, 2 }    -- sentence 2
      },
      { 4, 3 }  -- sentence lengths
   }   
}

local encoder = nn.Sequential()
   :add(nn.ParallelTable()
           :add(fwd_model)
           :add(bwd_model))
   :add(nn.JoinTable(2))
   :add(nn.Linear(2*hiddenSize, hiddenSize))

local decoder = nn.Sequential()
   :add(oxnn.SequenceOfWords{
           lookuptable = nn.Sequential():add(nn.LookupTable(vocabSize, 
                                                            hiddenSize))
              :add(nn.SplitTable(2)),
           recurrent = oxnn.ModelUtil.LSTMCell1(
              nn.Linear(hiddenSize, hiddenSize),
              nn.Linear(hiddenSize, hiddenSize),
              true),
           output = nn.Sequential()
              :add(nn.Linear(hiddenSize, vocabSize))
              :add(oxnn.LogSoftMaxInplace(true,true)),
           loss = 'nllloss'})
   :add(nn.SelectTable(2))
        
-- Forward and backward
local encoding = encoder:forward(input)
local decoderInput =    {
   { { encoding, torch.zeros(2,hiddenSize) } },
   torch.LongTensor{ 
      { 1, 7, 9, 2 },   -- sentence 1
      { 1, 3, 2, 2 }    -- sentence 2
   },
   { 4, 3 }  -- sentence lengths
}

local loss = decoder:forward(decoderInput)
encoder:backward(input, decoder:backward(decoderInput, 0)[1][1][1])

