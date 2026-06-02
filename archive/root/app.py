"""
OCD Pipeline — Terminal menu
Run: python app.py
"""
import subprocess
import sys


MENU = """
╔══════════════════════════════════╗
║       OCD Pipeline Menu          ║
╠══════════════════════════════════╣
║  資料生成                        ║
║   1. 生成參數組合 (params.py)    ║
║   2. 執行 RCWA 模擬              ║
╠══════════════════════════════════╣
║  訓練模型                        ║
║   3. 訓練 ANN                    ║
║   4. 訓練 CNN                    ║
╠══════════════════════════════════╣
║  預測 / 反推                     ║
║   5. ANN Forward（參數 → 光譜）  ║
║   6. ANN Inverse（光譜 → 參數）  ║
║   7. CNN Inverse（光譜 → 參數）  ║
╠══════════════════════════════════╣
║   0. 離開                        ║
╚══════════════════════════════════╝
"""


def run(cmd: list[str]) -> None:
    """執行子程序，繼承 terminal（顯示即時輸出）。"""
    subprocess.run([sys.executable] + cmd)


def ask_material() -> str:
    from data_generation.materials import MATERIALS
    names = list(MATERIALS.keys())
    print("\n可用材質：")
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name}")
    while True:
        raw = input("選擇材質編號（Enter = Si）: ").strip()
        if raw == "":
            return "Si"
        if raw.isdigit() and 1 <= int(raw) <= len(names):
            return names[int(raw) - 1]
        print("請輸入有效編號")


def main() -> None:
    while True:
        print(MENU)
        choice = input("請選擇功能: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            run(["data_generation/params.py"])

        elif choice == "2":
            material = ask_material()
            test = input("測試模式（只跑前 100 筆）? [y/N]: ").strip().lower() == "y"
            cmd = ["data_generation/generate.py", "--material", material]
            if test:
                cmd.append("--test")
            run(cmd)

        elif choice == "3":
            run(["ann/train.py"])

        elif choice == "4":
            run(["cnn/train.py"])

        elif choice == "5":
            run(["ann/predict.py"])

        elif choice == "6":
            run(["ann/inverse.py"])

        elif choice == "7":
            run(["cnn/predict.py"])

        else:
            print("無效選項，請重新輸入。")

        input("\n按 Enter 回到主選單...")


if __name__ == "__main__":
    main()
