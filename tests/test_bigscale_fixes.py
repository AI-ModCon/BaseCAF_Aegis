"""Tests for the two bigscale fixes:
  1. _popen_with_retry — retries on EAGAIN instead of crashing.
  2. instance.sh.j2 template — conda-unpack is inside the flock so concurrent
     instances on the same node cannot corrupt shared tokenizer files.
"""
import errno
import subprocess
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Fix 1: _popen_with_retry
# ---------------------------------------------------------------------------

from aegis.launcher import _popen_with_retry, _POPEN_MAX_RETRIES, _POPEN_RETRY_BASE


class TestPopenWithRetry:
    """_popen_with_retry should absorb EAGAIN and succeed on a later attempt."""

    def _eagain(self):
        err = BlockingIOError()
        err.errno = errno.EAGAIN
        return err

    def test_succeeds_on_first_try(self):
        proc = MagicMock(spec=subprocess.Popen)
        with patch("aegis.launcher.subprocess.Popen", return_value=proc) as mock_popen:
            result = _popen_with_retry(["echo", "hi"])
        mock_popen.assert_called_once()
        assert result is proc

    def test_retries_on_eagain_then_succeeds(self):
        proc = MagicMock(spec=subprocess.Popen)
        side_effects = [self._eagain(), self._eagain(), proc]
        with patch("aegis.launcher.subprocess.Popen", side_effect=side_effects) as mock_popen, \
             patch("aegis.launcher.time.sleep") as mock_sleep:
            result = _popen_with_retry(["echo", "hi"])

        assert mock_popen.call_count == 3
        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list[0] == call(_POPEN_RETRY_BASE * 1)
        assert mock_sleep.call_args_list[1] == call(_POPEN_RETRY_BASE * 2)
        assert result is proc

    def test_reraises_eagain_after_max_retries(self):
        with patch("aegis.launcher.subprocess.Popen", side_effect=self._eagain()), \
             patch("aegis.launcher.time.sleep"):
            with pytest.raises(BlockingIOError) as exc_info:
                _popen_with_retry(["echo", "hi"])
        assert exc_info.value.errno == errno.EAGAIN

    def test_non_eagain_oserror_reraises_immediately(self):
        err = OSError(errno.ENFILE, "too many open files")
        with patch("aegis.launcher.subprocess.Popen", side_effect=err), \
             patch("aegis.launcher.time.sleep") as mock_sleep:
            with pytest.raises(OSError) as exc_info:
                _popen_with_retry(["echo", "hi"])
        assert exc_info.value.errno == errno.ENFILE
        mock_sleep.assert_not_called()

    def test_other_exceptions_reraise_immediately(self):
        with patch("aegis.launcher.subprocess.Popen", side_effect=OSError("nope")), \
             patch("aegis.launcher.time.sleep") as mock_sleep:
            with pytest.raises(OSError, match="nope"):
                _popen_with_retry(["echo", "hi"])
        mock_sleep.assert_not_called()

    def test_max_retries_count_is_reasonable(self):
        assert 4 <= _POPEN_MAX_RETRIES <= 16

    def test_total_wait_covers_meaningful_backoff(self):
        total = sum(_POPEN_RETRY_BASE * (2 ** i) for i in range(_POPEN_MAX_RETRIES - 1))
        assert total >= 5.0, f"total backoff {total:.1f}s is too short to be useful"


# ---------------------------------------------------------------------------
# Fix 2: conda-unpack inside the flock in instance.sh.j2
# ---------------------------------------------------------------------------

def _render_template(extra_kwargs=None):
    from jinja2 import Environment, PackageLoader
    env = Environment(loader=PackageLoader("aegis", "templates"))
    template = env.get_template("instance.sh.j2")
    kwargs = dict(
        model="openai/gpt-oss-120b",
        tensor_parallel_size=4,
        port=8000,
        hf_home="/tmp/hf_home",
        extra_vllm_args=["--max-model-len", "131072"],
        conda_env="/home/user/assets/my_env.tar.gz",
        apptainer_image=None,
        ze_affinity_mask="0,1,2,3",
        modelinfo_cache_script="/tmp/dummy.py",
    )
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return template.render(**kwargs)


class TestCondaUnpackInFlock:
    """conda-unpack must run inside the flock so concurrent instances cannot
    corrupt shared tokenizer files via simultaneous in-place rewrites."""

    def test_conda_unpack_inside_flock_block(self):
        script = _render_template()
        flock_start = script.index("flock 9")
        # Find the closing paren of the flock subshell — sentinel is right after it
        sentinel_pos = script.index(".setup_complete\n  fi\n)")
        # Search for the full path to avoid matching occurrences in comments
        unpack_pos = script.index("/tmp/conda_env/bin/conda-unpack")
        assert flock_start < unpack_pos < sentinel_pos, (
            "conda-unpack must appear inside the flock subshell, before the sentinel touch"
        )

    def test_conda_unpack_not_called_after_flock(self):
        script = _render_template()
        flock_close = script.index(".setup_complete\n  fi\n)") + len(".setup_complete\n  fi\n)")
        after_flock = script[flock_close:]
        assert "conda-unpack" not in after_flock, (
            "conda-unpack must not appear outside the flock — that would allow concurrent runs"
        )

    def test_unpack_called_by_full_path(self):
        """Inside the flock subshell the env isn't activated, so conda-unpack
        must be invoked via its full path in /tmp/conda_env/bin/."""
        script = _render_template()
        assert "/tmp/conda_env/bin/conda-unpack" in script

    def test_source_activate_still_present_after_flock(self):
        """The parent shell still needs to activate the env after the flock."""
        script = _render_template()
        flock_close = script.index(".setup_complete\n  fi\n)") + len(".setup_complete\n  fi\n)")
        after_flock = script[flock_close:]
        assert "source /tmp/conda_env/bin/activate" in after_flock

    def test_setup_complete_sentinel_after_unpack(self):
        """Sentinel must be touched only after a successful unpack."""
        script = _render_template()
        unpack_pos = script.index("conda-unpack")
        sentinel_pos = script.index("touch /tmp/conda_env/.setup_complete")
        assert unpack_pos < sentinel_pos

    def test_no_bare_conda_unpack_without_conda_env(self):
        """Apptainer path has no conda env, so conda-unpack must not appear."""
        script = _render_template(extra_kwargs={"conda_env": None, "apptainer_image": "/img.sif"})
        assert "conda-unpack" not in script


# ---------------------------------------------------------------------------
# Fix 3: tiktoken pre-warm block serializes vocab cache population per node
# ---------------------------------------------------------------------------

class TestTiktokenPrewarm:
    """The tiktoken pre-warm block must serialize the first
    load_harmony_encoding() call per node so that concurrent vllm workers
    never race to write the same cache file."""

    def test_tiktoken_cache_dir_exported(self):
        script = _render_template()
        assert "export TIKTOKEN_CACHE_DIR=/tmp/tiktoken_cache" in script

    def test_prewarm_block_present(self):
        script = _render_template()
        assert "load_harmony_encoding" in script
        assert ".prewarm_complete" in script

    def test_prewarm_uses_separate_lock_fd(self):
        """Pre-warm must use a different fd than the conda extraction flock
        so the two serializations are independent."""
        script = _render_template()
        assert "flock 200" in script
        assert "tiktoken_prewarm.lock" in script

    def test_prewarm_after_conda_activation(self):
        """Pre-warm must appear after conda env is activated (so python and
        openai_harmony are available) but before vllm serve starts."""
        script = _render_template()
        activate_pos = script.index("source /tmp/conda_env/bin/activate")
        prewarm_pos = script.index("load_harmony_encoding")
        serve_pos = script.index("vllm serve")
        assert activate_pos < prewarm_pos < serve_pos, (
            "pre-warm must appear between conda activation and vllm serve"
        )

    def test_prewarm_does_not_suppress_errors(self):
        """The pre-warm block must not swallow errors with 2>/dev/null or
        || true — failures should be visible in job logs."""
        script = _render_template()
        # Find just the pre-warm block
        prewarm_start = script.index("flock 200")
        prewarm_end = script.index("tiktoken_prewarm.lock")
        prewarm_block = script[prewarm_start:prewarm_end]
        assert "2>/dev/null" not in prewarm_block
        assert "|| true" not in prewarm_block

    def test_no_prewarm_without_conda_env(self):
        """Apptainer path has no conda env, so pre-warm must not appear."""
        script = _render_template(extra_kwargs={"conda_env": None, "apptainer_image": "/img.sif"})
        assert "TIKTOKEN_CACHE_DIR" not in script
        assert "prewarm_complete" not in script
