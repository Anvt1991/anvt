# Script khởi động bot bằng lệnh: python -m app.main
# File này đặt ở thư mục gốc (cùng cấp với thư mục app)

if __name__ == "__main__":
    import runpy
    runpy.run_module("app.main", run_name="__main__") 