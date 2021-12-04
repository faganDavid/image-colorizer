# How to setup the environment to run this app

Install Ubuntu for WSL2 from the Windows Store: https://www.microsoft.com/en-us/p/ubuntu/9nblggh4msv6?activetab=pivot:overviewtab
Install VS Code on Windows.

Open the Ubunutu Terminal and run 'code .' from the flask workspace folder (the folder where you have this flask app)

VS Code will now be connected to Ubuntu WSL2 allowing you to use linux commands from the terminal in there.

Follow this tutorial for further information: https://code.visualstudio.com/docs/python/tutorial-flask

Steps to get the app running are outlined below:

Install repo for python3.7 installation
    sudo add-apt-repository ppa:deadsnakes/ppa
Install virtualenv command to create virtual environment for the flask app
    sudo apt install virtualenv

Check your python version before installing python3.7
    python -V
Install python3.7, 3.7 is what is needed for tensorflow version 1
    sudo apt install python3.7 -y

Make sure your original python version wasn't overridden by installing python3.7 (if it was you'll have to change the path back)
    python -V

Check bin folder to make sure you see python3.7
    ls /usr/bin/py*

Create the virtual environment to run the flask app in and install all the dependencies
    virtualenv --python=/usr/bin/python3.7 venv

Activate the environment
    source venv/bin/activate

Install the needed dependencies inside the virtualenvironment 
Make sure you see venv on the left of your shell to know the environmnent is activated first.
    pip install -r requirements.txt

Set the python interpreter in VS Code to run the flask app:
    CTRL+SHFT+P
    Python: Select interpreter
    Enter interpreter path
    Find...
        Choose the director: image-colorizor/venv/bin/python3.7
            The location for mine was: /mnt/f/workspace/image-colorizer/venv/bin/