#!/bin/bash

SIZE=$1
FILE=$2

echo $SIZE > $FILE
for i in $(seq 1 $SIZE)
do
  val=$(printf "%03d.%03d%02d" $(( $RANDOM % 1000 )) $(( $RANDOM % 1000 )) $(( $RANDOM % 100)))
  echo $val >> $FILE
done
