#!/bin/bash

if [ -z "$1" ]
then
    for file in ./.ci_support/*.yaml;
    do
        ./build-locally.py $(basename $file .yaml);
    done
else
    for F in $(cat $1);
    do
        ./build-locally.py $F;
    done
fi


