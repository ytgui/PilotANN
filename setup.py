import torch
from setuptools import setup, Command
from torch.utils import cpp_extension


class TestCommand(Command):
    description = "test"
    user_options = []

    def initialize_options(self):
        assert torch.cuda.is_available()

    def finalize_options(self):
        pass

    def run(self):
        import pytest
        pytest.main([
            '--cov=pilot_ann', '--tb=long', 'test/'
        ])


setup(
    name='PilotANN',
    version='1.0.0',
    packages=['pilot_ann'],
    ext_modules=[
        cpp_extension.CUDAExtension(
            'pilot_ann.extension',
            sources=[
                'extension/entry.cpp',
                'extension/ops_cpu.cpp',
                'extension/sampling_cpu.cpp'
            ],
            extra_compile_args={
                'cxx': [
                    '-mavx2', '-mfma', '-fopenmp',
                    '-ffast-math', '-funroll-loops'
                ]
            }
        )
    ],
    cmdclass={
        'test': TestCommand,
        'build_ext': cpp_extension.BuildExtension,
    },
    install_requires=[
    ]
)
