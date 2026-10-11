#!/bin/bash
#SBATCH --job-name=bc-launch
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/scratch/pandya.kei/simfoundry-repro/logs/bc_launch_%j.out
# Runs jobs/bluecube_campaign.sh ON THE CLUSTER, so the whole campaign --
# submission, the rollouts, the dependent finisher and the render -- is
# independent of any interactive shell. Submit this and close the terminal.
set -uo pipefail
bash /scratch/pandya.kei/simfoundry-repro/jobs/bluecube_campaign.sh \
     "${N:-6}" "${SCENE:-shift}" "${STEPS:-700}"
