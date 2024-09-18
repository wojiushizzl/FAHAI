# FAHAI don't know AI

This is a Flet app that can help to uses the YOLO model to develop your own AI vision project

![FAHAI](./component/img.png)
## Usage
To run the app:

```
python3 develop.py # for development
python3 deploy.py # for deployment
```

## Installation
To install the app:

```
git clone https://github.com/wojiushizzl/FAHAI.git
```
Create a virtual environment
```
python3 -m venv venv python=python3.8
source venv/bin/activate
pip install -r requirements.txt
```
If your have a NVIDIA GPU, you can install the GPU version of the below packages
```
# install this pageage is not a easy job, you can refer to the official website
pip install torch==2.2.1+cu118 torchvision==0.17.1+cu118 -i https://download.pytorch.org/whl/torch_stable.html
```
this project has been tested on the following environment:
```
python==3.8
windows10
windows11
macos
unbuntu 20.04
```

## Further Development
If you want to publish this flet project to multiple platform, you can refer to the flet official website
https://flet.dev/docs/publish



## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
```markdown
MIT License

Copyright (c) [year] [fullname]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```