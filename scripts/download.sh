#!/bin/sh

set -e

CURDIR=$(pwd)
echo $CURDIR



if curl -L -o ~/Downloads/mnist-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset > /dev/null; then
	echo "Download Successful";
else
	echo "Downdload Failed";
fi

