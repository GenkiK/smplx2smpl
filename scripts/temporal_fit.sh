export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

PROJ_DIR="$(dirname $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd))"
cd ${PROJ_DIR}
source .env

CUDA_VISIBLE_DEVICES=$1 python temporal_fit.py \
    --root_smplx_dir ${DATA_DIR}/smplx_seq \
    --root_per_frame_result_dir ${DATA_DIR}/smpl_seq_interp_smoothed \
    --root_temporal_result_dir ${DATA_DIR}/smpl_seq_interp_smoothed_temporal \
    --session_name s1 \
    --do_visualize \
    --root_save_img_dir ${DATA_DIR}/smpl_seq_interp_smoothed_temporal_vis \
    --smooth_pose_vel_weight 1000 \
    --smooth_pose_acc_weight 1000 \
    --smooth_joints_vel_weight 3000 \
    --smooth_joints_acc_weight 3000 \
    --smooth_verts_vel_weight 3000 \
    --smooth_verts_acc_weight 3000 \
    --smooth_transl_vel_weight 5000 \
    --smooth_transl_acc_weight 5000 \
    --consider_gender