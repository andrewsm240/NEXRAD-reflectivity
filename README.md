# Plugin for NEXRAD reflectivity

This is a simple plugin downloads NEXRAD files from Amazon S3 and extracts vertical profiles of base reflectivity above a selected latitude and longitude  

## Overview

Plugins contain both code and packaging information. In this example, we've organized them as follows:

1. The code consists of:
    * [main.py](./main.py). Main plugin code. It's primarily structured around the `process_frame` function.
    * [test.py](./test.py). Minimal test file which exercises `process_frame` on a test image. Serves as a starting point for building automated testing.
    * [requirements.txt](./requirements.txt). Python dependencies file. Add any required modules to this file.

2. The packaging information consists of:
    * [sage.yaml](./sage.yaml). Defines plugin info used by [ECR](https://portal.sagecontinuum.org). You must update this for your example.
    * [Dockerfile](./Dockerfile). Defines plugin code and dependency bundle. You can update this if you have additional dependencies not covered by [requirements.txt](./requirements.txt).
