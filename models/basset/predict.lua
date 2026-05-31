-- MultiMolecule
-- Copyright (C) 2024-Present  MultiMolecule

-- This file is part of MultiMolecule.

-- MultiMolecule is free software: you can redistribute it and/or modify
-- it under the terms of the GNU Affero General Public License as published by
-- the Free Software Foundation, either version 3 of the License, or
-- any later version.

-- MultiMolecule is distributed in the hope that it will be useful,
-- but WITHOUT ANY WARRANTY; without even the implied warranty of
-- MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
-- GNU Affero General Public License for more details.

-- You should have received a copy of the GNU Affero General Public License
-- along with this program.  If not, see <http://www.gnu.org/licenses/>.

-- For additional terms and clarifications, please refer to our License FAQ at:
-- <https://multimolecule.danling.org/about/license-faq>.

require 'torch'

local model_file = arg[1]
local sequence = string.upper(arg[2])
local out_file = arg[3]
local basset_src = arg[4]

package.path = basset_src .. '/?.lua;' .. package.path
cuda = false
cuda_nn = false
require 'convnet'

local alphabet = {A = 1, C = 2, G = 3, T = 4}
local input = torch.Tensor(1, 4, 1, string.len(sequence)):zero()
for i = 1, string.len(sequence) do
    local base = string.sub(sequence, i, i)
    local channel = alphabet[base]
    if channel == nil then
        error('Unsupported DNA base: ' .. base)
    end
    input[{1, channel, 1, i}] = 1
end

local convnet = ConvNet:__init()
convnet:load(torch.load(model_file))
convnet.model:evaluate()
convnet.model:forward(input)

local logits = convnet.model.modules[#convnet.model.modules - 1].output:float()
local handle = io.open(out_file, 'w')
for target = 1, logits:size(2) do
    if target > 1 then
        handle:write('\t')
    end
    handle:write(string.format('%.9g', logits[{1, target}]))
end
handle:write('\n')
handle:close()
