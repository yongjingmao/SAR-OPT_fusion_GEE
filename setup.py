import setuptools

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

setuptools.setup(
    name = "SAR-OPT_fusion_GEE",
    version = "0.0.1",
    author = "Yongjing Mao",
    author_email = "maomao940405@gmail.com",
    description = "A GEE based python package to characterize and infill gaps of optical satellite images",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/yongjingmao/SAR-OPT_fusion_GEE",
    package_dir = {"": "scripts"},
    packages = setuptools.find_packages(include=["scripts", "scripts.*"]),
    python_requires = ">=3.7",
    install_requires = [
        "earthengine-api == 0.1.270"
        "geopandas == 0.8.1",
		"pandas == 1.1.0",
		"json == 2.0.9"
    ]
)