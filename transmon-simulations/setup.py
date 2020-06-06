import setuptools

setuptools.setup(
    name="transmon_simulations",
    py_modules=["single_transmon", "transmon_chain", "two_transmons"],
    version="0.1.2",
    author="Gleb Fedorov",
    author_email="vdrhtc@gmail.com",
    description="Transmon simulations module",
    long_description="Classes for simulations of coupled transmon systems",
    long_description_content_type="text/markdown",
    url="https://github.com/vdrhtc/Examples/transmon_simulations",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
