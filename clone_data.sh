#!/bin/sh

mv data data_buf
mkdir data
scp -r pi@192.168.43.175:/home/pi/77-car-camera/data data
rm -rf data_buf
