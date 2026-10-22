# The wheel holds one shared library for one platform, so it is not "pure Python" and carries a platform tag
# (py3-none-<platform>). Build one per platform with scripts/build_wheels.py.
from setuptools import setup

try:
    from wheel.bdist_wheel import bdist_wheel

    class PlatformWheel(bdist_wheel):
        def finalize_options(self):
            super().finalize_options()
            self.root_is_pure = False

        def get_tag(self):
            _, _, plat = super().get_tag()
            return "py3", "none", plat

    setup(cmdclass={"bdist_wheel": PlatformWheel})
except ImportError:  # building an sdist without wheel installed
    setup()
