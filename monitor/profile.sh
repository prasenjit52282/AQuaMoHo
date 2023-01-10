#!bin/bash
# Usage: bash profile.sh exp_name track_code
# Example: bash profile.sh "python3 test.py"
echo "$0 $1  $2"
expname=$1
monitorcode=$2
now=$(date +"%m_%d_%Y_%H_%M_%S")

echo "TEMP,PID,USER,PR,NI,VIRT,RES,SHR,S,CPU,MEM,TIME,ENV,COMMAND" >> "exp_${expname}_${now}.txt"

while :
do
    {
     vcgencmd measure_temp | tr -s 'temp=' 'T' | tr -s "'C\n" 'C';
     top -c -b -n 1 | grep "$monitorcode" | grep -wv "bash\|grep" | tr -s ' ' ','
    } >> "exp_${expname}_${now}.txt"
    sleep 1
    echo $(date)
done