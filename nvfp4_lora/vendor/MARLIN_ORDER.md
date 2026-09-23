# Marlin token-order research patch

`marlin_order.patch` contains the exact Python-module patch from
[vLLM draft PR #52532](https://github.com/vllm-project/vllm/pull/52532), authored
by bxchange, pinned at commit
`e8a07dcccf8e48dd4b9b42a355fc6d5b9db59073`. Its motivating report is
[vLLM issue #52525](https://github.com/vllm-project/vllm/issues/52525).
The draft was unmerged when this research patch was prepared.

Copyright contributors to the vLLM project. The patch and the original source
fixture `tests/fixtures/vllm_marlin_moe_0_27_1.py.txt` are Apache-2.0 licensed;
the license is included in `LICENSE` beside this file. The original module's
SPDX notices remain intact. The fixture is the installed module from the exact
base image identified by `reef_marlin_patch.MARLIN_BASE_IMAGE_ID`.

The patch only inserts the upstream canonicalization function and its guarded
call before the existing Marlin expert GEMMs. The installer refuses other base
source bytes, repeated patching, changed patch assets, or a different output.
Its manifest's `helper_sha256` covers the canonicalization function source,
from its `def` line through its final statement with a terminating newline;
it does not refer to the installer itself.

This remains a diagnostic hypothesis for the pinned Nemotron NVFP4 actor.
It does not establish global batch invariance, qualify a different model,
or change the numerical acceptance criteria. The production image is untouched;
the operations build creates a separate derived research image.
