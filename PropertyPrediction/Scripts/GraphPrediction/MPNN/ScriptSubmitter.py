import os
import subprocess

# Directory containing the generated training scripts
scripts_dir = 'generated_scripts'
submission_dir = 'submission_scripts'

# Create a directory to store the submission scripts
os.makedirs(submission_dir, exist_ok=True)

# List all the training scripts in the directory
training_scripts = [f for f in os.listdir(scripts_dir) if f.endswith('.py')]

# Generate the SCRIPTS variable dynamically
scripts_variable = 'SCRIPTS=(' + ' '.join(f'"{script}"' for script in training_scripts) + ')'

# Define the base submission script template with placeholders
submission_template = f"""
#!/bin/bash
#PBS -l select=1:ncpus=1:mem=64gb:ngpus=1
#PBS -l walltime=72:00:00
#PBS -J 1-{len(training_scripts)}
#PBS -N Transformer_array_job

# Change to the directory where the scripts are located
cd {os.path.abspath(scripts_dir)}

# Array of script names corresponding to the datasets
{scripts_variable}

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate gt4sd

# Run the Python script corresponding to the PBS_ARRAY_INDEX
python3 ${SCRIPTS[$PBS_ARRAY_INDEX-1]}
"""

# Create a single submission script
submission_filename = os.path.join(submission_dir, 'submit_array_job.sh')
with open(submission_filename, 'w') as submission_file:
    submission_file.write(submission_template)

# Submit the job script
result = subprocess.run(['qsub', submission_filename], capture_output=True, text=True)
if result.returncode != 0:
    print(f"Error submitting {submission_filename}: {result.stderr}")
else:
    print(f"Successfully submitted {submission_filename}: {result.stdout}")
