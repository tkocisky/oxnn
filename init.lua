require 'torch'
require 'nn'
require 'optim'
require 'lfs'
require 'paths'
require 'xlua'
u = require 'moses'
require 'nnx'
require 'paths'
require 'xlua'
require 'nngraph'

--------------------------------------------------------------------------------
-------------------------[[ oxnn -- Oxford NN Library. ]]-----------------------
--------------------------------------------------------------------------------
oxnn = {}

--[[ util ]]--
torch.include('oxnn', 'util/ModelUtil.lua')
torch.include('oxnn', 'util/Optimizer.lua')
torch.include('oxnn', 'util/cloneManyTimes.lua')
torch.include('oxnn', 'util/util.lua')

--[[ nn modification ]]--
-- torch.include('oxnn', 'nn_modif/LookupTable.lua')
torch.include('oxnn', 'nn_modif/MM.lua')
torch.include('oxnn', 'nn_modif/SelectTable.lua')

--[[ nn ]]--
torch.include('oxnn', 'nn/CAddTableNoCopy.lua')
torch.include('oxnn', 'nn/CAddTableNoCopyInplace.lua')
torch.include('oxnn', 'nn/CMulTable2.lua')
torch.include('oxnn', 'nn/CriterionTable.lua')
torch.include('oxnn', 'nn/Index.lua')
torch.include('oxnn', 'nn/JoinTable.lua')
torch.include('oxnn', 'nn/LSTM12Part2.lua')
torch.include('oxnn', 'nn/LinearBlockDiagonal.lua')
torch.include('oxnn', 'nn/LinearCAddInplace.lua')
torch.include('oxnn', 'nn/LinearNoOutputZero.lua')
torch.include('oxnn', 'nn/LogSoftMaxInplace.lua')
torch.include('oxnn', 'nn/NarrowTable.lua')
torch.include('oxnn', 'nn/NoAccGradParameters.lua')
torch.include('oxnn', 'nn/VecsToVecs.lua')
torch.include('oxnn', 'nn/ZeroLoss.lua')

--[[ nngraph modification ]]--
torch.include('oxnn', 'nngraph_modif/gmodule.lua')

--[[ rnn ]]--
torch.include('oxnn', 'rnn/ClassNLLCriterionMasked.lua')
torch.include('oxnn', 'rnn/Constant.lua')
torch.include('oxnn', 'rnn/Mask.lua')
torch.include('oxnn', 'rnn/MaskedAdd.lua')
torch.include('oxnn', 'rnn/RPUtil.lua')
torch.include('oxnn', 'rnn/RecurrentPropagator.lua')
torch.include('oxnn', 'rnn/SequenceOfWords.lua')
torch.include('oxnn', 'rnn/SumLosses.lua')

--[[ text ]]--
torch.include('oxnn', 'text/TextUtil.lua')
torch.include('oxnn', 'text/Vocabulary.lua')

--[[ test ]]--
torch.include('oxnn', 'test/test.lua')

--[[ cuda ]]--
torch.include('oxnn', 'util/CudaUtil.lua')
oxnn.cu = oxnn.CudaUtil(false)
function oxnn.InitCuda()
   require 'cutorch'
   require 'cunn'
   --require 'cunnx'
   pcall(require, 'cudnn')
   pcall(require, 'fbcunn')
   torch.include('oxnn', 'nn_modif/LookupTableGPU.lua')
   oxnn.cu = oxnn.CudaUtil(true)
   require 'liboxnn'
end


return oxnn
