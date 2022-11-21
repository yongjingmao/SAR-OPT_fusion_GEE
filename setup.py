import setuptools

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

setuptools.setup(
    name = "optical_sar_blending",
    version = "0.0.1",
    author = "Yongjing Mao",
    author_email = "maomao940405@gmail.com",
    description = "A GEE based python package to characterize and infill gaps of optical satellite images",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://bitbucket.csiro.au/projects/DIG/repos/optical_sar_blending",
    package_dir = {"": "optical_sar_blending"},
    packages = setuptools.find_packages(include=["optical_sar_blending", "optical_sar_blending.*"]),
    python_requires = ">=3.7",
    install_requires = [
        "earthengine-api == 0.1.270"
        "geojson == 2.5.0"
    ]
)