import setuptools

setuptools.setup(
    name="NeurGO",  # 修改包名为 NeurGO
    version="1.0.0",
    author="Anonymous",  
    author_email="anonymous@example.com", 
    description="NeurGO: A Generative MetaBBO Framework",  # 补充简短描述
    long_description="Implementation of NeurGO", 
    long_description_content_type="text",
    url="",  
    packages=setuptools.find_packages(),  
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)