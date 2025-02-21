for inst in ../instances/queen*; do
	trueInst=${inst:13}
	jobName=q${trueInst:5:-4}_$1
	echo $trueInst 
	echo $jobName
	sbatch --job-name=$jobName --output=$1/${trueInst:0:-4}.job_out run_job.sh $trueInst $1
done
