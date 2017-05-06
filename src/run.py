# -*- encoding:utf-8 -*-
import subprocess

cmd = 'python3 train_test.py'
subprocess.call(cmd, shell=True)

cmd = 'python3 train_validation.py'
subprocess.call(cmd, shell=True)

cmd = 'python3 lgb.py'
subprocess.call(cmd, shell=True)
