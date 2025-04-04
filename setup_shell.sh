# Make sure we have the conda environment set up.
CONDA_PATH=~/miniconda3/bin/activate
ENV_NAME=ipl
REPO_PATH=path/to/repository
USE_MUJOCO_PY=true
WANDB_API_KEY="b98d2b806f364f5af900550ec98e26e2f418e8a7" # If you want to use wandb, set this to your API key.

# Setup Conda
source $CONDA_PATH
conda activate $ENV_NAME
cd $REPO_PATH
unset DISPLAY # Make sure display is not set or it will prevent scripts from running in headless mode.

if $WANDB_API_KEY; then
    export WANDB_API_KEY=$"b98d2b806f364f5af900550ec98e26e2f418e8a7"
fi

if $USE_MUJOCO_PY; then
    echo "Using mujoco_py"
    if [ -d "/usr/lib/nvidia" ]; then
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
    fi
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin
fi

# First check if we have a GPU available
if nvidia-smi | grep "CUDA Version"; then
    if [ -d "/usr/local/cuda-11.7" ]; then # This is the only GPU version supported by compile.
        export PATH=/usr/local/cuda-11.7/bin:$PATH
    elif [ -d "/usr/local/cuda" ]; then
        export PATH=/usr/local/cuda/bin:$PATH
        echo "Using default CUDA. Compatibility should be verified. torch.compile requires >= 11.7"
    else
        echo "Warning: Could not find a CUDA version but GPU was found."
    fi
    export MUJOCO_GL="egl"
    # Setup any GPU specific flags
else
    echo "GPU was not found, assuming CPU setup."
    export MUJOCO_GL="osmesa" # glfw doesn't support headless rendering
fi

export D4RL_SUPPRESS_IMPORT_ERROR=1
# If you want to use a different location for D4RL, set this flag:
# export D4RL_DATASET_DIR="path/to/d4rl"
