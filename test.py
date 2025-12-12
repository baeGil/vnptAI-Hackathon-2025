import sympy as sp

# Định nghĩa ma trận
ma_tran = sp.Matrix([[1, 0, 0], [2, 1, 0], [0, 2, 1]])

# Tìm các giá trị riêng
gia_tri_rieng = ma_tran.eigenvals()

print("Các giá trị riêng của toán tử tuyến tính T là:", gia_tri_rieng)