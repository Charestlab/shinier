import subprocess
import sys
import textwrap

import pytest


pytestmark = pytest.mark.unit_tests


def test_import_falls_back_when_optional_cconvolve_fails():
    code = textwrap.dedent(
        """
        import importlib.abc
        import importlib.machinery
        import sys
        import warnings

        class BrokenCconvolveFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "shinier._cconvolve":
                    return importlib.machinery.ModuleSpec(fullname, BrokenCconvolveLoader())
                return None

        class BrokenCconvolveLoader(importlib.abc.Loader):
            def create_module(self, spec):
                return None

            def exec_module(self, module):
                raise ImportError("simulated NumPy ABI mismatch")

        sys.meta_path.insert(0, BrokenCconvolveFinder())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import shinier

        assert shinier._HAS_CYTHON is False
        assert shinier.convolve2d_direct is None
        assert shinier.convolve2d_separable is None
        assert any(
            "optional compiled convolution extension" in str(warning.message)
            and "slower NumPy fallback" in str(warning.message)
            and "Python:" in str(warning.message)
            and "NumPy:" in str(warning.message)
            and "reinstalling SHINIER" in str(warning.message)
            and "simulated NumPy ABI mismatch" in str(warning.message)
            for warning in caught
        )
        """
    )

    result = subprocess.run(
        [sys.executable, "-W", "always", "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
