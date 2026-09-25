"""Pinned inputs. Every digest here was read from its source, none is invented."""

BASE_REPO = "google/diffusiongemma-26B-A4B-it"
BASE_REVISION = "f7f5b7f5fa82ffc52addd066915886d497f5517b"

# The base weights and config, as a manifest (the initial champion).
BASE_FILES = {
    "config.json": "13b11d2fe87302cc2332c64eb9eb4ac305d9b8a123ffe9c5cb5b1920fc70c506",
    "model.safetensors.index.json": (
        "6e33e8465d55fe6c7bc0a5453c7a4b341e6467d032c6ded82aaf439f61dac69a"
    ),
    "model-00001-of-00011.safetensors": (
        "3efe137998af7d2bde4e3ab04ab3524823699a4ac3130adace5003ef40cceeb6"
    ),
    "model-00002-of-00011.safetensors": (
        "4a39d68c756fb26bbd2a54f2b8d550047ea98f3152f87ec75db825c8e17934a7"
    ),
    "model-00003-of-00011.safetensors": (
        "ac6083e3489215ca032501714b78832e5cc4c945a8dbbb905b5292a4e95bc75e"
    ),
    "model-00004-of-00011.safetensors": (
        "865b66393de5a9c752e67beae1bd2c860c12786b72ad86ca9345c00b7b586e60"
    ),
    "model-00005-of-00011.safetensors": (
        "a87e01bed77ad9d2234851267d99af583ff79c7a86d7494fe08ae0f9de9cd318"
    ),
    "model-00006-of-00011.safetensors": (
        "077e841b3b138fbc38df2c36665abdaafa96b8c96f968e2263a7452afaa912ab"
    ),
    "model-00007-of-00011.safetensors": (
        "13a18b3c04a7f19a16385dd8a0d7fd0b9cc89ef131e4294a063c31c7d90c58ef"
    ),
    "model-00008-of-00011.safetensors": (
        "aca5d3bdfc84700bd55b475781cb46e5608104c4832d2abe111a119ddcb23ff7"
    ),
    "model-00009-of-00011.safetensors": (
        "8e2418867354e5cb356c0af8ccfdcc60bc363500674d6cf395991e7d6219eb29"
    ),
    "model-00010-of-00011.safetensors": (
        "93d564b7dd686464a5c068ff9665cd5d3bca399c2ce320aecd41bd011e3787d5"
    ),
    "model-00011-of-00011.safetensors": (
        "afec047176bb2a05f078566576aec6bdb71ad4d041275d0d0c89473fda6d6d87"
    ),
}

# Tokenizer, chat template and processor files always come from the base snapshot.
BASE_SUPPORT_FILES = {
    "chat_template.jinja": "9aeb7eac68ad87bba7567e9d4597ff203e5609f1b427d9e823437d0142cc61bf",
    "generation_config.json": "99334f763c3dbe8b161aeaca1c150a05344299fda2d2e4a0e1d342c744461200",
    "model_index.json": "989c453462f9b84b90f02fe82b89f23c743b513a767a1789292da76f056f5117",
    "processor_config.json": "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c",
    "scheduler/scheduler_config.json": (
        "5e5536ea036c284bcc09c7b08045868d741a41a3090069f6c035697bba31c6c7"
    ),
    "tokenizer.json": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    "tokenizer_config.json": "a284d1243b62be31faa9c13e1c28cece940c4abaa7bd9ad87b94f61b40687200",
}

# vLLM nightly with PR 57250 (structured reads); commit 7f1a5398 is 77 commits past the merge.
VLLM_IMAGE = (
    "vllm/vllm-openai:nightly-7f1a5398e9610d96c473931a26c0e12bbe0d0423"
    "@sha256:ed3c505d2cf4b62b0ca8d0b87591bcb6fbff2a3abcc6d25732343e19b4d74f09"
)
STRUCTURED_SERVER_URL = (
    "https://raw.githubusercontent.com/vllm-project/vllm/"
    "1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8/"
    "examples/features/structured_diffusion/structured_server.py"
)
STRUCTURED_SERVER_SHA256 = "7cd9aa0081090c064eaac28db0f54f812749eeb3ae787d7f7653d2e35d8a938f"

# The runtime lane measures NVFP4 weights only, on B300 only (docs/operator.md, runtime lane).
# The reference export is NVIDIA's ModelOpt NVFP4 of the same base: W4A4 FP4 (e2m1 in U8 pairs,
# group 16, F8_E4M3 block scales, F32 global and input scales) on the 128 routed experts of the
# 30 layers; attention, the dense MLP, the router, embeddings, lm_head and self-conditioning stay
# BF16 (its hf_quant_config.json). Digests read from the Hub at this revision (lfs sha256 or a
# download of the file).
NVFP4_REPO = "nvidia/diffusiongemma-26B-A4B-it-NVFP4"
NVFP4_REVISION = "ec4ff3df205028f4e81c954c2227f9312b3ec2ea"
NVFP4_FILES = {
    "config.json": "b4f650bd55f6c6ccd55656e27a7967c72ac28f1f3e315edaa29d6f5eebd05fde",
    "model.safetensors.index.json": (
        "3a1f1d8b64daad31b33c63ac48edd1cb43f0c3b3edf7cddf625f74fff37ae4c0"
    ),
    "model-00001-of-00002.safetensors": (
        "24f1257810848d296cc42fd2f6186c8dd6c2a562482c4c1bc8463a087a2c6554"
    ),
    "model-00002-of-00002.safetensors": (
        "5015731988fbfc21042032ed9ea4c9af335bfe70bcd7529d09b8f2438d5fdd0e"
    ),
}
# The base config.json plus its quantization_config (modelopt, NVFP4, group 16, an FP8 KV
# scheme without k/v scales). The runtime lane pins --kv-cache-dtype bfloat16: `auto` would
# resolve to FP8 with unit scales. Its chat_template.jinja differs from the base's, so the base
# support files are served with it, as with every quality champion.
NVFP4_CONFIG_SHA256 = NVFP4_FILES["config.json"]
# sha256 of the canonical {tensor name: [dtype, shape]} of both shard headers (47,067 tensors):
# a runtime model must have exactly this layout, so FP4 packing, block scales and global scales
# are where the loader expects them. Values are not checked (a scale can still be wrong; the
# fidelity reads catch what matters).
NVFP4_SCHEMA_SHA256 = "ac30a8c9d59750901192c3856988fdbf8b3c081e50561c9b953c4de2b5933d7c"
