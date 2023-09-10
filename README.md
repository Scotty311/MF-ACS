The project contains a `requirements.txt` file that we can use to install the following dependencies into your 
virtual environment:

```bash
opencv_python_headless==4.5.2.54
streamlit==0.82.0
mediapipe==0.8.4.2
numpy==1.18.5
Pillow==8.2.0
```

```bash
pip install -r requirements.txt
```

### StreamLit

[Streamlit](https://docs.streamlit.io) is an open-source Python library that makes it easy to create and share beautiful, custom web apps for 
machine learning and data science. In just a few minutes you can build and deploy powerful data apps.

Create a new Python file `face_mesh_app.py` and import the dependencies:

```python
import streamlit as st
import mediapipe as mp
import cv2 as cv
import numpy as np
import tempfile
import time
from PIL import Image
```

Test your installation by running the following and opening your browser on `localhost:8501`:

```bash
streamlit run face_checkin_app.py
```
