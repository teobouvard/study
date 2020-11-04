- Number of unique mac adresses

tshark -r Downloads/Vatican1/probes-2013-02-24.pcap1 -T fields -e wlan.sa | uniq | wc -l
