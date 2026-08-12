"""
The test that guards the entire purpose of this package.

``manyfews_core`` exists so the ManyFEWS science can run in a Colab notebook, in
CI, or in a browser-adjacent build script without PostGIS, GDAL, GeoDjango,
celery or an LLVM toolchain. That property is easy to lose to a single
convenience import, and nothing else would notice until someone tried to install
it somewhere minimal.

Runs in a subprocess because the rest of the suite may legitimately have imported
these already.
"""

import subprocess
import sys
import textwrap

FORBIDDEN = [
    "django",
    "celery",
    "numba",
    "llvmlite",
    "osgeo",
    "shapely",
    "psycopg2",
    "geopandas",
    "rasterio",
    "tenacity",
]


def test_importing_the_package_pulls_in_nothing_heavy():
    script = textwrap.dedent(
        f"""
        import sys
        import manyfews_core            # noqa: F401

        forbidden = {FORBIDDEN!r}
        found = sorted(
            name for name in forbidden
            if name in sys.modules or any(m.startswith(name + ".") for m in sys.modules)
        )
        if found:
            raise SystemExit("heavy dependencies imported: " + ", ".join(found))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_full_pipeline_imports_stay_light():
    """The public API, not just the top-level module."""
    script = textwrap.dedent(
        f"""
        import sys
        from manyfews_core import (
            FloodEmulator, load_parameters, run_ensemble, spin_up,
            rasterise, inject_storm, cached_channel_mask,
        )

        forbidden = {FORBIDDEN!r}
        found = sorted(
            name for name in forbidden
            if name in sys.modules or any(m.startswith(name + ".") for m in sys.modules)
        )
        if found:
            raise SystemExit("heavy dependencies imported: " + ", ".join(found))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_optional_extras_are_not_imported_eagerly():
    """
    matplotlib and folium are optional. Importing the package must not need them,
    so a headless build script installs two dependencies rather than twenty.
    """
    script = textwrap.dedent(
        """
        import sys
        import manyfews_core            # noqa: F401

        eager = [n for n in ("matplotlib", "folium", "PIL") if n in sys.modules]
        if eager:
            raise SystemExit("optional extras imported eagerly: " + ", ".join(eager))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
