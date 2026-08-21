"""The serve flags are shell, but a wrong flag costs a rented boot — so the
selection logic is exercised directly."""
import subprocess

SCRIPT = "runner-vllm/onstart.sh"


def _precision_flag(precision: str) -> str:
    """Run the script's flag-selection block in isolation."""
    body = open(SCRIPT).read()
    start = body.index("# precision-flag-begin")
    end = body.index("# precision-flag-end")
    snippet = body[start:end]
    out = subprocess.run(
        ["bash", "-c", f'PRECISION="{precision}"\n{snippet}\necho "$PRECISION_FLAG"'],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def test_a_dtype_becomes_dtype_not_quantization():
    """--quantization bfloat16 is not a valid vLLM argument; the server would
    refuse to start and the rented boot would be wasted."""
    assert _precision_flag("bfloat16") == "--dtype bfloat16"


def test_float16_is_also_a_dtype():
    assert _precision_flag("float16") == "--dtype float16"


def test_auto_is_a_dtype():
    assert _precision_flag("auto") == "--dtype auto"


def test_a_quantisation_scheme_stays_quantization():
    assert _precision_flag("fp8") == "--quantization fp8"


def test_awq_stays_quantization():
    assert _precision_flag("awq") == "--quantization awq"
