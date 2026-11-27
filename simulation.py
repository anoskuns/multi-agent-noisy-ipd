import random
import numpy as np
import matplotlib.pyplot as plt

# Cố định seed để chạy lại luôn ra kết quả giống nhau
random.seed(42)
np.random.seed(42)

# Quy ước: 0 là Hợp tác, 1 là Phản bội
HOP_TAC = 0
PHAN_BOI = 1

# Bảng tính điểm: (hành_động_A, hành_động_B) -> (điểm_A, điểm_B)
BANG_DIEM = {
    (HOP_TAC, HOP_TAC): (3, 3),
    (HOP_TAC, PHAN_BOI): (0, 5),
    (PHAN_BOI, HOP_TAC): (5, 0),
    (PHAN_BOI, PHAN_BOI): (1, 1),
}

# ==========================================
# 1. CÁCH CHỌN NƯỚC ĐI CỦA TỪNG CHIẾN LƯỢC
# ==========================================

def chon_nuoc_di(ten_chien_luoc, nuoc_doi_thu_truoc, nuoc_minh_truoc, bang_q):
    # 1. Kẻ ích kỷ: Lúc nào cũng phản bội
    if ten_chien_luoc == "Always-Defect":
        return PHAN_BOI

    # 2. Ăn miếng trả miếng (Tit-for-Tat): Vòng đầu hợp tác, sau đó bắt chước đối thủ
    elif ten_chien_luoc == "Tit-for-Tat (TFT)":
        if nuoc_doi_thu_truoc is None:
            return HOP_TAC
        return nuoc_doi_thu_truoc

    # 3. Người vị tha (Generous TFT): Giống TFT nhưng có 20% tha thứ nếu bị phản bội
    elif ten_chien_luoc == "Generous TFT (p=0.2)":
        if nuoc_doi_thu_truoc is None:
            return HOP_TAC
        if nuoc_doi_thu_truoc == PHAN_BOI:
            if random.random() < 0.2:  # 20% cơ hội bỏ qua
                return HOP_TAC
            return PHAN_BOI
        return HOP_TAC

    # 4. Tự học (Q-Learning): Tra bảng điểm kinh nghiệm (bang_q) để chọn
    elif ten_chien_luoc == "Q-Learning (Self-Play)":
        # Trạng thái hiện tại là cặp (nước của mình vòng trước, nước đối thủ vòng trước)
        trang_thai = (nuoc_minh_truoc, nuoc_doi_thu_truoc)
        if trang_thai not in bang_q:
            bang_q[trang_thai] = [0.0, 0.0]  # [điểm kỳ vọng nếu Hợp tác, điểm kỳ vọng nếu Phản bội]

        # 10% đi bừa để khám phá, 90% chọn nước có điểm cao hơn trong bảng Q
        if random.random() < 0.1:
            return random.choice([HOP_TAC, PHAN_BOI])
        
        if bang_q[trang_thai][HOP_TAC] >= bang_q[trang_thai][PHAN_BOI]:
            return HOP_TAC
        else:
            return PHAN_BOI


# ==========================================
# 2. HÀM ĐẤU 1 TRẬN (100 VÒNG)
# ==========================================

def lam_nhieu_tin_hieu(hanh_dong, ti_le_nhieu):
    """Mô phỏng nhìn nhầm: có xác suất bị lật ngược hành động"""
    if random.random() < ti_le_nhieu:
        return 1 - hanh_dong  # 0 thành 1, 1 thành 0
    return hanh_dong

def dau_mot_tran(ten_chien_luoc, ti_le_nhieu, bang_q_A, bang_q_B, so_vong=100):
    # Lưu lại ký ức vòng trước (ban đầu chưa có gì nên là None)
    A_nhin_thay_B = None
    B_nhin_thay_A = None
    nuoc_A_truoc = None
    nuoc_B_truoc = None

    so_lan_hop_tac = 0
    tong_diem_hai_ben = 0

    for _ in range(so_vong):
        # Bước 1: Hai bên chọn nước đi
        nuoc_A = chon_nuoc_di(ten_chien_luoc, A_nhin_thay_B, nuoc_A_truoc, bang_q_A)
        nuoc_B = chon_nuoc_di(ten_chien_luoc, B_nhin_thay_A, nuoc_B_truoc, bang_q_B)

        # Bước 2: Tính điểm thật
        diem_A, diem_B = BANG_DIEM[(nuoc_A, nuoc_B)]
        tong_diem_hai_ben += (diem_A + diem_B)
        
        if nuoc_A == HOP_TAC:
            so_lan_hop_tac += 1
        if nuoc_B == HOP_TAC:
            so_lan_hop_tac += 1

        # Bước 3: Tín hiệu truyền qua môi trường có nhiễu
        A_nhin_thay_B_moi = lam_nhieu_tin_hieu(nuoc_B, ti_le_nhieu)
        B_nhin_thay_A_moi = lam_nhieu_tin_hieu(nuoc_A, ti_le_nhieu)

        # Bước 4: Nếu là Q-Learning thì cập nhật rút kinh nghiệm vào bảng Q
        if ten_chien_luoc == "Q-Learning (Self-Play)":
            # Cập nhật cho A
            tt_cu_A = (nuoc_A_truoc, A_nhin_thay_B)
            tt_moi_A = (nuoc_A, A_nhin_thay_B_moi)
            if tt_moi_A not in bang_q_A:
                bang_q_A[tt_moi_A] = [0.0, 0.0]
            # Công thức Q-learning: Q_cũ + tốc_độ_học * (điểm_nhận_được + 0.95 * max_Q_tương_lai - Q_cũ)
            bang_q_A[tt_cu_A][nuoc_A] += 0.1 * (diem_A + 0.95 * max(bang_q_A[tt_moi_A]) - bang_q_A[tt_cu_A][nuoc_A])

            # Cập nhật cho B
            tt_cu_B = (nuoc_B_truoc, B_nhin_thay_A)
            tt_moi_B = (nuoc_B, B_nhin_thay_A_moi)
            if tt_moi_B not in bang_q_B:
                bang_q_B[tt_moi_B] = [0.0, 0.0]
            bang_q_B[tt_cu_B][nuoc_B] += 0.1 * (diem_B + 0.95 * max(bang_q_B[tt_moi_B]) - bang_q_B[tt_cu_B][nuoc_B])

        # Bước 5: Ghi nhớ để dùng cho vòng sau
        A_nhin_thay_B = A_nhin_thay_B_moi
        B_nhin_thay_A = B_nhin_thay_A_moi
        nuoc_A_truoc = nuoc_A
        nuoc_B_truoc = nuoc_B

    ti_le_hop_tac = (so_lan_hop_tac / (2 * so_vong)) * 100
    return ti_le_hop_tac, tong_diem_hai_ben


# ==========================================
# 3. CHẠY THÍ NGHIỆM & VẼ BIỂU ĐỒ
# ==========================================

cac_chien_luoc = [
    "Always-Defect",
    "Tit-for-Tat (TFT)",
    "Generous TFT (p=0.2)",
    "Q-Learning (Self-Play)"
]
cac_muc_nhieu = [0.0, 0.05, 0.10, 0.20]  # 0%, 5%, 10%, 20%
so_tran_dau = 500

ket_qua_hop_tac = {}
ket_qua_diem = {}

print("=== KẾT QUẢ THỰC NGHIỆM ===\n")

for ten in cac_chien_luoc:
    print(f"--- Chiến lược: {ten} ---")
    ket_qua_hop_tac[ten] = []
    ket_qua_diem[ten] = []

    for nhieu in cac_muc_nhieu:
        # Tạo bảng Q mới cho mỗi mức nhiễu (giữ nguyên xuyên suốt 500 trận để AI tích lũy kinh nghiệm)
        bang_q_A = {}
        bang_q_B = {}
        
        ds_hop_tac = []
        ds_diem = []

        for _ in range(so_tran_dau):
            ht, diem = dau_mot_tran(ten, nhieu, bang_q_A, bang_q_B)
            ds_hop_tac.append(ht)
            ds_diem.append(diem)

        tb_hop_tac = np.mean(ds_hop_tac)
        do_lech = np.std(ds_hop_tac)
        tb_diem = np.mean(ds_diem)

        ket_qua_hop_tac[ten].append(tb_hop_tac)
        ket_qua_diem[ten].append(tb_diem)

        print(f"  Nhiễu {int(nhieu*100):2d}% | Hợp tác: {tb_hop_tac:5.1f}% (±{do_lech:4.1f}%) | Điểm TB: {tb_diem:5.1f}/600")
    print()

# Vẽ 2 biểu đồ cạnh nhau
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
nhan_truc_x = ["0%", "5%", "10%", "20%"]
ky_hieu = ['x', 'o', 's', '^']

for ten, kh in zip(cac_chien_luoc, ky_hieu):
    ax1.plot(nhan_truc_x, ket_qua_hop_tac[ten], marker=kh, linewidth=2, label=ten)
    ax2.plot(nhan_truc_x, ket_qua_diem[ten], marker=kh, linewidth=2, label=ten)

ax1.set_title("Tỷ lệ hợp tác trung bình theo mức nhiễu")
ax1.set_xlabel("Mức nhiễu quan sát")
ax1.set_ylabel("Tỷ lệ hợp tác (%)")
ax1.set_ylim(-5, 105)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

ax2.set_title("Tổng điểm phúc lợi theo mức nhiễu")
ax2.set_xlabel("Mức nhiễu quan sát")
ax2.set_ylabel("Tổng điểm 2 bên (Tối đa = 600)")
ax2.set_ylim(150, 620)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.savefig("results1.png", dpi=300)
plt.show()
print("Đã lưu biểu đồ thành công vào file 'results.png'!")