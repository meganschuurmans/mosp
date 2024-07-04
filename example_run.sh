#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=62:00:00
#SBATCH --qos=low
#SBATCH --container-mounts=share to mount for data location \
#SBATCH --container-image="location of your container"

python3 /path/to/local/repository/mosp/main.py                                      --mosp_dir="/path/to/local/repository/mosp/" \
                                                                                    --archive_overview_dir="/path/to/overview/excel/training/data.xlsx" \
                                                                                    --overview_dir="/path/to/folder/training_images/and/image/with/label/overview/" \
                                                                                    --external_dir="/path/to/external/dataset/" \
                                                                                    --multimodal_dir="/path/to/folder/with/multimodal_models/" \
                                                                                    --imaging_model_dir="/path/to/folder/with/imaging_models/" \
                                                                                    --clinical_model_dir="/path/to/folder/with/clinical_models/" \
                                                                                    --results_dir="/path/to/folder/to/save/results/" \
                                                                                    --model_phase="inference" \
                                                                                    --modal_type="mm" 
                                                                                    # --ensemble 
                                                                                    # --ema \
                                                                                    # --early_ensemble \
                                                                                    # --double_ensemble \
                                                                                    # --ensemble 