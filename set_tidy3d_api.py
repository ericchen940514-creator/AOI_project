import tidy3d as td
import tidy3d.web as web  # 強制匯入 web 模組

# 1. 在這裡貼上你的 API Key
your_api_key = "把這串換成你的_API_KEY"

try:
    # 使用強制匯入後的 web 模組進行配置
    web.configure(api_key=your_api_key)
    print("\n✅ Tidy3D API Key 已成功設定！")
    
    # 測試是否能連線
    print(f"目前帳號狀態: {web.get_info()}")
    
except Exception as e:
    print(f"\n❌ 還是失敗了，錯誤訊息: {e}")
    print("\n💡 建議測試：在終端機輸入 'pip show tidy3d' 看看版本號是多少。")