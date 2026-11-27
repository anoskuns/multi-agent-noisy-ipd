import random
import numpy as np
import matplotlib.pyplot as plt

# Cố định seed để tái lập kết quả
random.seed(42)
np.random.seed(42)

HOP_TAC = 0
PHAN_BOI = 1

BANG_DIEM = {
    (HOP_TAC, HOP_TAC): (3, 3),
    (HOP_TAC, PHAN_BOI): (0, 5),
    (PHAN_BOI, HOP_TAC): (5, 0),
    (PHAN_BOI, PHAN_BOI): (1, 1),
}

# ==========================================
# 1. HÀM CHỌN NƯỚC ĐI & LÀM NHIỄU
# ==========================================

def nuoc_di_gtft(nuoc_doi_thu_truoc, ti_le_tha_thu):
    """Generous TFT với xác suất tha thứ p truyền vào"""
    if nuoc_doi_thu_truoc is None:
        return HOP_TAC
    if nuoc_doi_thu_truoc == PHAN_BOI:
        if random.random() < ti_le_tha_thu:
            return HOP_TAC
        return PHAN_BOI
    return HOP_TAC

def lam_nhieu_tin_hieu(hanh_dong, ti_le_nhieu):
    if random.random() < ti_le_nhieu:
        return 1 - hanh_dong
    return hanh_dong


# ==========================================
# 2. HÀM ĐẤU 1 TRẬN CHO 2 KỊCH BẢN
# ==========================================

def dau_self_play(ti_le_tha_thu, ti_le_nhieu, so_vong=100):
    """Kịch bản A: GTFT(p) đấu với GTFT(p) -> Trả về tổng điểm 2 bên (tối đa 600)"""
    A_thay_B = None
    B_thay_A = None
    tong_diem = 0

    for _ in range(so_vong):
        nuoc_A = nuoc_di_gtft(A_thay_B, ti_le_tha_thu)
        nuoc_B = nuoc_di_gtft(B_thay_A, ti_le_tha_thu)

        diem_A, diem_B = BANG_DIEM[(nuoc_A, nuoc_B)]
        tong_diem += (diem_A + diem_B)

        A_thay_B = lam_nhieu_tin_hieu(nuoc_B, ti_le_nhieu)
        B_thay_A = lam_nhieu_tin_hieu(nuoc_A, ti_le_nhieu)

    return tong_diem

def dau_voi_always_defect(ti_le_tha_thu, ti_le_nhieu, so_vong=100):
    """Kịch bản B: GTFT(p) đấu với Always-Defect -> Trả về điểm RIÊNG của GTFT (tối đa 100)"""
    GTFT_thay_doi_thu = None
    diem_rieng_gtft = 0

    for _ in range(so_vong):
        nuoc_gtft = nuoc_di_gtft(GTFT_thay_doi_thu, ti_le_tha_thu)
        nuoc_dich = PHAN_BOI  # Always-Defect luôn phản bội

        diem_A, _ = BANG_DIEM[(nuoc_gtft, nuoc_dich)]
        diem_rieng_gtft += diem_A

        GTFT_thay_doi_thu = lam_nhieu_tin_hieu(nuoc_dich, ti_le_nhieu)

    return diem_rieng_gtft


# ==========================================
# 3. CHẠY THÍ NGHIỆM 2 & VẼ BIỂU ĐỒ
# ==========================================

cac_muc_p = [0.01, 0.20, 0.40, 0.90]
nhieu_co_dinh = 0.10   # Cố định nhiễu 10%
so_tran_dau = 500

ket_qua_self_play = []
ket_qua_vs_defect = []

print(f"=== KẾT QUẢ THÍ NGHIỆM 2: ABLATION STUDY TRÊN GTFT (NHIỄU {int(nhieu_co_dinh*100)}%) ===\n")

for p in cac_muc_p:
    ds_self = [dau_self_play(p, nhieu_co_dinh) for _ in range(so_tran_dau)]
    ds_adv = [dau_voi_always_defect(p, nhieu_co_dinh) for _ in range(so_tran_dau)]

    tb_self = np.mean(ds_self)
    tb_adv = np.mean(ds_adv)

    ket_qua_self_play.append(tb_self)
    ket_qua_vs_defect.append(tb_adv)

    print(f"  p = {p:4.2f} | Self-play: {tb_self:5.1f}/600 | vs. Always-Defect (điểm GTFT): {tb_adv:5.1f}/100")

# Vẽ 2 biểu đồ cột cạnh nhau để thấy rõ sự đánh đổi (Trade-off)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
nhan_x = [f"p = {p}" for p in cac_muc_p]
mau_sac = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

# Biểu đồ 1: Self-play
cot1 = ax1.bar(nhan_x, ket_qua_self_play, color=mau_sac, width=0.55)
ax1.set_title("Kịch bản A: Self-Play (Chống nhiễu 10%)")
ax1.set_xlabel("Xác suất tha thứ (p)")
ax1.set_ylabel("Tổng điểm 2 bên (Tối đa = 600)")
ax1.set_ylim(0, 650)
ax1.grid(axis='y', linestyle='--', alpha=0.6)
ax1.bar_label(cot1, fmt='%.1f', padding=3)

# Biểu đồ 2: vs Always-Defect
cot2 = ax2.bar(nhan_x, ket_qua_vs_defect, color=mau_sac, width=0.55)
ax2.set_title("Kịch bản B: vs. Always-Defect (Khả năng tự vệ)")
ax2.set_xlabel("Xác suất tha thứ (p)")
ax2.set_ylabel("Điểm riêng của GTFT (Tối đa = 100)")
ax2.set_ylim(0, 115)
ax2.grid(axis='y', linestyle='--', alpha=0.6)
ax2.bar_label(cot2, fmt='%.1f', padding=3)

plt.tight_layout()
plt.savefig("results2.png", dpi=300)
plt.show()
print("\nĐã lưu biểu đồ thành công vào file 'ablation_gtft.png'!")