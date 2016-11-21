#!/bin/bash
kill `ps aux | grep job_worker_connect.py | awk '{print $2}'`
