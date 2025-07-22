export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

PROJ_DIR="$(dirname $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd))"
cd ${PROJ_DIR}
source .env

# SMPL-X
CUDA_VISIBLE_DEVICES=$1 python visualize.py \
    --root_smplx_dir ${DATA_DIR}/smplx_seq \
    --root_save_img_dir ${DATA_DIR}/smplx_vis \
    --session_name s1

# SMPL
CUDA_VISIBLE_DEVICES=$1 python visualize.py \
    --root_smplx_dir ${DATA_DIR}/smplx_seq \
    --root_src_dir ${DATA_DIR}/smpl_seq_interp_smoothed_temporal \
    --root_save_img_dir ${DATA_DIR}/smpl_final_vis \
    --session_name s1