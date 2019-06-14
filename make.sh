#!/bin/bash
cd model/correlation_package
python setup.py install --user
cd ../roialign_package
python setup.py install --user