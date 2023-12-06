from setuptools import setup, Extension
from Cython.Build import cythonize


try:
    extensions = [
        Extension("ctools_weights",
                sources=["src/distance_based_weighting.pyx"],
                # include_dirs=["../src/include"],
                language="c++"
                )
    ]
    setup(
        name='weight_algorithms',
        ext_modules=cythonize(
            extensions, 
            compiler_directives={'language_level': 3},
            build_dir="build",
            )
    )
    
    import subprocess
    # subprocess.call(["cython","-a",cython_module])

except Exception as e:
    print(e)
    input("Press Enter to close...")
    raise e