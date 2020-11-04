#!/bin/bash
input_dir=~/Downloads/Vatican1/*
output="UUID.csv"


for pcap in $input_dir
do
    echo "Extracting $pcap"
    tshark -r $pcap -Y "wps.uuid_e != \"\"" -T fields -E separator=, -e wlan.sa -e wps.uuid_e | sort | uniq >> $output
done
