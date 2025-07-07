# RNA3D-SSCL
RNA3D-SSCL: Improving RNA Tertiary Structure Prediction via a Secondary Structure-Constrained Loss Function


# Preparation
1. Linux systems
   
3. Python

python                    3.8.11

numpy                     1.20.3

scipy                     1.7.1

pytorch                   1.9.0

3. OpenMM (https://openmm.org/)



# Add new training data
If you want to add new training data, Each RNA consists of four files, for example:

8G4I_1_g.fasta

8G4I_1_gss.npy

8G4I_1_g_coor.npy

8G4I_1_g.pdb

They include the sequence information of RNA, the secondary structure information of RNA, the positions of key atoms in RNA, and the original pdb file.
Utilize two  tools for secondary structure prediction: RNAfold and EternaFold.

The predicted secondary structure information is provided in the form of a pairing matrix and pairing probability graph. 



# Use
Modify the relevant configurations in config.py
## Train model
Adjust the relevant settings and run  "python train.py"

Training Parameters

msa_dim

Specifies the number of multiple sequence alignment (MSA) input features.

m_dim, s_dim, z_dim

Dimensions for the model's internal representations:

N_ensemble

Number of ensemble passes used during structure prediction (useful for stochastic averaging).

N_cycle

Number of iterative  cycles in the prediction module.

wordcwd

Working directory of the current project. (Please modify as needed)

data_path

Path to the training/validation dataset. (Please modify as needed)

save_model_path

Directory where trained model checkpoints will be saved. (Please modify as needed)

model_name

Filename or tag used to identify the saved model.

device

GPU device to use. Change if multiple GPUs or CPU-only mode is desired.

seed

Random seed to ensure reproducibility of results.

max_len

Maximum sequence length supported by the model.

lr

Initial learning rate for the optimizer.

T_max

Maximum number of iterations (epochs) for cosine annealing learning rate scheduling(Please modify as needed).

save_fren

Frequency (in epochs) at which to save model checkpoints(Please modify as needed).

ara_fren

Frequency (in epochs) at which to run Arena evaluation or analysis(Please modify as needed).



## Predict and refine
Adjust the relevant settings, load the corresponding model and run  "python predict_refine.py"
Predict and refine Parameters

ARENA_BIN_PATH

Path to the external Arena tool binary. Make sure this path points to a valid installation.

checkpoint_path

Path to a checkpoint file for resuming training or evaluation.

pre_in_dir

Input directory for prediction mode.

pre_out_dir

Output directory for saving prediction results.



# Citation
Jingwei Lu, Yifan Wu, Yiming Li, Qianpei Liu, Fuhao Zhang, Min Li, and Min Zeng.
RNA3D-SSCL: Improving RNA Tertiary Structure Prediction via a Secondary Structure-Constrained Loss Function
