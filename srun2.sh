#!/bin/sh


# Load the singularity module
module load singularity

echo "Running on $(hostname)"

# Execute the commands within the Singularity container
singularity exec --nv /home/mtk01/cs/xtuner/nemo_2401.sif bash -c "
    source env/bin/activate
    bash rec_run_3.sh
"&

# Run nvidia-smi in parallel
while true; do
    nvidia-smi
    sleep 60  # Adjust the sleep interval as needed
done &
 
# Wait for the rec_run_0.sh script to finish
wait


#SBATCH --job-name=run_nemo
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu

# srun --pty --nodes=16 --ntasks-per-node=1 --nodelist=hgpn034,hgpn035,hgpn036,hgpn037,hgpn038,hgpn039,hgpn040,hgpn041,hgpn042,hgpn043,hgpn044,hgpn045,hgpn046,hgpn[090-092] --gres=gpu:8 --cpus-per-task=8 --partition=gp1d bash ./srun.sh