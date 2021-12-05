# How to setup the environment to run this app locally to do development and local testing.

Install VS Code on Windows.

Set VS Code default terminal to anaconda CMD with the conda environment you have
the tensorflow libraries in to do the colorization, for me that is 'py37'.

You can change the default Terminal back when you are finished.

Install the needed dependencies
    pip install -r requirements.txt

Set the python interpreter in VS Code to run the flask app:
    CTRL+SHFT+P
    Python: Select interpreter
    Enter interpreter path
    Find...
        Choose the anaconda directory with python3.7: ~\anaconda3\envs\py37\python.exe

The 'Run and Debug' section should allow you to create a launch.json file.

Run the flask app from VS Code (F5).