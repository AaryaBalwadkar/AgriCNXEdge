# #!/bin/bash
# echo "====================================================="
# echo " STARTING AUTOMATED ABLATION SUITE"
# echo "====================================================="

# # Ablation A: No Distillation (Hard Labels Only)
# echo "[1/4] Running Ablation A (No Distillation)..."
# python run_student.py \
#     --output-dir outputs/ablation_A_no_kd \
#     --alpha 1.0 \
#     --beta 0.0

# # Ablation B: Static Task Weighting (No Homoscedastic)
# echo "[2/4] Running Ablation B (Static Weights)..."
# python run_student.py \
#     --output-dir outputs/ablation_B_static_weights \
#     --disable-homo

# # Ablation C: No LoRA (Frozen backbone stages 2,3)
# echo "[3/4] Running Ablation C (No LoRA)..."
# python run_student.py \
#     --output-dir outputs/ablation_C_no_lora \
#     --no-lora

# # Ablation D: No Phase 1 Warm-up
# echo "[4/4] Running Ablation D (No Warm-up)..."
# python run_student.py \
#     --output-dir outputs/ablation_D_no_warmup \
#     --freeze-ep 0

# echo "====================================================="
# echo " ALL ABLATION STUDIES COMPLETED"
# echo "====================================================="

#!/bin/bash
echo "====================================================="
echo " STARTING AUTOMATED ABLATION SUITE"
echo "====================================================="

# Ablation B: Static Task Weighting (No Homoscedastic)
echo "[2/4] Running Ablation B (Static Weights)..."
python run_student.py \
    --output-dir outputs/ablation_B_static_weights \
    --disable-homo

echo "====================================================="
echo " ALL ABLATION STUDIES COMPLETED"
echo "====================================================="