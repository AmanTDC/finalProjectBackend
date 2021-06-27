#!/bin/bash
sudo apt update
sudo apt install npm
sudo npm -g install serverless
sudo apt install python3-virtualenv
sudo apt install python3-pip
sudo apt install docker
sudo snap install docker
git clone "https://github.com/AmanTDC/finalProjectBackend"
cd final*
npm init -f
npm install --save-dev serverless-wsgi serverless-python-requirements
virtualenv venv --python=python3
source venv/bin/activate
pip install boto3
pip install numpy
pip install flask==1.1.3
pip install torch 
pip install opencv-python
pip freeze>requirements.txt
sls