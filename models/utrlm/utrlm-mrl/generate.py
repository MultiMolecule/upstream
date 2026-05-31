# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This file is part of MultiMolecule.

# MultiMolecule is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# MultiMolecule is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://multimolecule.danling.org/about/license-faq>.

"""Generate the UTR-LM MRL checkpoint-parity fixture from official code."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
TE_EL_GENERATOR = HERE.parents[1] / "utrlm-te_el" / "generate.py"
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("utrlm-te_el-generator", TE_EL_GENERATOR)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to import {TE_EL_GENERATOR}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

module.configure_case(
    case="utrlm-mrl",
    variant="MRL",
    checkpoint_env="MULTIMOLECULE_UPSTREAM_UTRLM_MRL_CHECKPOINT",
    checkpoint_official_rel_path=(
        "Model/Pretrained/"
        "ESM2SISS_FS4.1_fiveSpeciesCao_6layers_16heads_128embedsize_"
        "4096batchToks_lr1e-05_supervisedweight1.0_structureweight1.0_"
        "MLMLossMin_epoch93.pkl"
    ),
    checkpoint_sha256="705ea278849702e12285d4059dc15d902cc445f458729415ed83b1bb6515f3d3",
    description=__doc__,
)

if __name__ == "__main__":
    module.main()
