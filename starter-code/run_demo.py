import json
from template import ChatbotBaseline, ToolCallingAgent

def main():
    print("======================================================================")
    print("VINUNI CODELAB DAY 04 — TEST CHUẨN ĐẦY ĐỦ 3 VÒNG LẶP (REACT LOOP)")
    print("======================================================================\n")

    # Kịch bản 1: Test mẫu mặc định của bài Lab (1 Tool)
    q1 = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."
    print(f"🔹 KỊCH BẢN 1 (Single Tool): '{q1}'")
    agent = ToolCallingAgent(max_iterations=5)
    r1 = agent.run(q1)
    print("\n👉 Final Answer:")
    print(r1["answer"])
    print("\n👉 Trace Log:")
    print(json.dumps(r1["trace"], indent=2, ensure_ascii=False))

    print("\n" + "="*70 + "\n")

    # Kịch bản 2: Gọi chuỗi cả 2 Tool (3 Vòng lặp giống slide minh họa của BTC)
    q2 = "Cho tôi xem resort Vinpearl giá dưới 6 triệu. Và tôi cũng muốn ghi nhận phản hồi: tên tôi là Phạm Thị Dung, phòng tôi đặt ở Vinpearl Nha Trang bị ẩm mốc, mức độ trung bình."
    print(f"🔹 KỊCH BẢN 2 (Chuỗi 2 Tool - 3 Iterations): '{q2}'")
    r2 = agent.run(q2)
    print("\n👉 Final Answer:")
    print(r2["answer"])
    print("\n👉 Trace Log (Đầy đủ iteration 1, 2, 3 như slide):")
    print(json.dumps(r2["trace"], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

