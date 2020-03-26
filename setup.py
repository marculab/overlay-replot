from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need
# fine tuning.
buildOptions = dict(packages = [], excludes = [])

base = 'Console'

executables = [
    Executable('replot.py', base=base)
]

includes = ['scipy.sparse.csgraph._validation', 'numpy']

setup(name='replot',
      version = '1.0',
      description = '',
      options = dict(build_exe = buildOptions),
      executables = executables)
