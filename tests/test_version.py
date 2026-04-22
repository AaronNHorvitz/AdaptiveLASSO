from importlib.metadata import version

from adaptive_lasso import __version__


def test_runtime_version_matches_distribution_metadata():
    assert version("adaptive-lasso") == __version__
