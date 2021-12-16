#!/bin/bash
/home/trent/miniconda3/envs/py37/bin/python /home/trent/miniconda3/envs/py37/bin/gunicorn -w 2 --worker-class gevent -b :80 app:app --daemon
