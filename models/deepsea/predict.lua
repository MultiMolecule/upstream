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
require 'nn'

pcall(require, 'cutorch')
pcall(require, 'cunn')

local model_file = arg[1]
local sequence = string.upper(arg[2])
local out_file = arg[3]

local width = string.len(sequence)
local alphabet = {A = 1, G = 2, C = 3, T = 4}
local complement = {A = 'T', C = 'G', G = 'C', T = 'A'}

local function is_module(object)
    local name = torch.typename(object)
    return name ~= nil and string.match(name, '^nn%.') ~= nil
end

local function unwrap_model(object)
    if is_module(object) then
        return object
    end
    if type(object) == 'table' then
        local keys = {'model', 'net', 'network', 'module'}
        for _, key in ipairs(keys) do
            if object[key] ~= nil then
                return unwrap_model(object[key])
            end
        end
    end
    error('Unsupported DeepSEA checkpoint payload: expected an nn module or table containing one')
end

local function encode(reverse_complement)
    local input = torch.FloatTensor(1, 4, width, 1):zero()
    for i = 1, width do
        local source_index = i
        if reverse_complement then
            source_index = width - i + 1
        end
        local base = string.sub(sequence, source_index, source_index)
        if reverse_complement then
            base = complement[base]
        end
        local channel = alphabet[base]
        if channel == nil then
            error('Unsupported DNA base: ' .. tostring(base))
        end
        input[{1, channel, i, 1}] = 1
    end
    return input
end

local function as_row(tensor)
    local output = tensor:float()
    if output:nDimension() == 1 then
        output = output:view(1, output:size(1))
    end
    if output:nDimension() ~= 2 or output:size(1) ~= 1 then
        error('Unexpected DeepSEA output shape: ' .. tostring(output:size()))
    end
    return output
end

local function logits_from_probabilities(probabilities)
    local logits = torch.FloatTensor(probabilities:size(2))
    for target = 1, probabilities:size(2) do
        local p = probabilities[{1, target}]
        if p < 1e-7 then
            p = 1e-7
        elseif p > 1.0 - 1e-7 then
            p = 1.0 - 1e-7
        end
        logits[target] = math.log(p / (1.0 - p))
    end
    return logits
end

local model = unwrap_model(torch.load(model_file))
model:float()
model:evaluate()

local forward = as_row(model:forward(encode(false))):clone()
local reverse = as_row(model:forward(encode(true))):clone()
local probabilities = (forward + reverse) / 2
local logits = logits_from_probabilities(probabilities)

local handle = io.open(out_file, 'w')
for target = 1, logits:size(1) do
    if target > 1 then
        handle:write('\t')
    end
    handle:write(string.format('%.9g', logits[target]))
end
handle:write('\n')
handle:close()
