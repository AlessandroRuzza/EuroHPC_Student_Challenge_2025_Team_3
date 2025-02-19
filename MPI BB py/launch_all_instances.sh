for inst in ../instances/*; do
	trueInst=${inst:13}
	jobName=${trueInst:0:-4}_$1
	echo $trueInst 
	echo $jobName
	sbatch --job-name=$jobName --output=$1/$jobName.job_out run_job.sh $trueInst $1
done
