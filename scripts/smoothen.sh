export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

PROJ_DIR="$(dirname $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd))"
cd ${PROJ_DIR}
source .env

CUDA_VISIBLE_DEVICES=$1 python smoothen.py \
    --root_src_dir ${DATA_DIR}/smpl_seq_interp \
    --root_tgt_dir ${DATA_DIR}/smpl_seq_interp_smoothed \
    --session_name s1