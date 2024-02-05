from setuptools import setup, Extension
from Cython.Build import cythonize

annotate = False

try:
    extensions = [
        Extension("ctools_weights",
                sources=["src/distance_based_weighting.pyx"],
                language="c++"
                ),
    ]
    setup(
        name='weight_algorithms',
        ext_modules=cythonize(
            extensions, 
            compiler_directives={'language_level': 3},
            build_dir="build",
            )
    )
    
    if annotate: 
        ext = extensions[0]
        out_name = ext.name
        src_file_path = ext.sources[0] # .pyx file
        build_dir = "build"
        
        import subprocess
        subprocess.call(["cython","-a", "-o", f"{build_dir}/{out_name}.cpp", src_file_path])
        
        # Move file
        import shutil
        shutil.move(f"{build_dir}/{out_name}.html", f"{out_name}.html")
        
        # Open with default browser
        import webbrowser
        webbrowser.open(f"{out_name}.html")

except Exception as e:
    print(e)
    input("Press Enter to close...")
    raise e