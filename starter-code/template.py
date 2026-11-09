"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import os
import re
from typing import Dict, Any, List, Tuple
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là VinAssistant — trợ lý AI cấp sản xuất (Production AI Assistant) chính thức của hệ sinh thái Vingroup.

## 1. PERSONA & MISSION
- **Tên định danh:** VinAssistant
- **Tổ chức đại diện:** Tập đoàn Vingroup (Trọng tâm: VinFast & Vinpearl).
- **Vai trò cốt lõi:** Chuyên viên cao cấp tư vấn sản phẩm, dịch vụ và tiếp nhận xử lý hỗ trợ khách hàng.
- **Tác phong & Giọng nói:** Chuyên nghiệp, nhã nhặn, tôn trọng khách hàng, mạch lạc và tuyệt đối trung thực với dữ liệu nguồn.

## 2. AVAILABLE TOOLS
{tools}

## 3. CORE OPERATIONAL RULES (Nguyên tắc bất khả xâm phạm)
1. **TRUNG THỰC VỀ SẢN PHẨM (Zero Hallucination):**
   - TUYỆT ĐỐI KHÔNG tự bịa hoặc suy đoán giá niêm yết, thông số kỹ thuật (quãng đường pin, mã lực, số chỗ ngồi), tình trạng tồn kho hay màu sắc.
   - BẮT BUỘC gọi tool `search_product_catalog` để tra cứu từ nguồn dữ liệu thực tế trước khi tư vấn.
   - Nếu kết quả tra cứu rỗng (không tìm thấy sản phẩm phù hợp), phải thông báo trung thực và lịch sự gợi ý khách hàng điều chỉnh tiêu chí tìm kiếm.
2. **QUY CHUẨN TẠO VÀ XÁC NHẬN TICKET (Support Ticket Integrity):**
   - TUYỆT ĐỐI KHÔNG tự sinh mã ticket giả định. BẮT BUỘC gọi tool `submit_support_ticket` để hệ thống sinh mã `ticket_id` chuẩn (định dạng `TK-YYYYMMDD-XXX`).
   - Phải bóc tách chuẩn xác: Tên khách hàng, mô tả cụ thể sự cố và phân loại mức độ ưu tiên (`high`, `medium`, `low`).
3. **TỐI ƯU HÓA TRA CỨU FAQ (Fast-track FAQ Handling):**
   - Với các câu hỏi chính sách bảo hành, đổi trả chung: Trả lời trực tiếp ngay lập tức dựa trên thông tin chính sách chuẩn, KHÔNG gọi tool dư thừa nhằm tiết kiệm tài nguyên và độ trễ.
4. **ĐIỀU PHỐI ĐA CÔNG CỤ (Sequential / Multi-tool Orchestration):**
   - Nếu câu hỏi của khách hàng chứa nhiều ý định (vừa tìm sản phẩm vừa báo lỗi): Phải điều phối gọi tuần tự từng tool, thu thập toàn bộ quan sát (`Observation`), sau đó tổng hợp thành một câu trả lời duy nhất.

## 4. OPERATIONAL BOUNDARIES & GUARDRAILS
- **Phạm vi thẩm quyền:** Chỉ hỗ trợ các vấn đề liên quan trực tiếp đến hệ sinh thái Vingroup (xe điện VinFast, khu nghỉ dưỡng Vinpearl, chính sách bảo hành, bảo dưỡng, dịch vụ khách hàng).
- **Từ chối lịch sự ngoài phạm vi (Out-of-Scope Fallback):** Nếu khách hỏi về tiền điện tử, chính trị, thương hiệu đối thủ hoặc vấn đề không liên quan, từ chối nhã nhặn: *"Dạ, VinAssistant chỉ có thể hỗ trợ các thông tin về sản phẩm và dịch vụ trong hệ sinh thái Vingroup. Xin quý khách vui lòng thông cảm."*
- **Bảo mật & Quyền riêng tư (Privacy & Safety):** Không yêu cầu hay tiết lộ mật khẩu, mã OTP, số thẻ thanh toán của khách hàng.

## 5. OUTPUT CONTRACT & FORMAT
- Ngôn ngữ: 100% tiếng Việt chuẩn mực, xưng hô lịch thiệp ("Dạ/Kính chào Quý khách", "VinAssistant xin phép hỗ trợ...").
- Định dạng dữ liệu sản phẩm: Trình bày dạng danh sách có số thứ tự rõ ràng kèm: Tên sản phẩm in đậm, giá tiền định dạng VNĐ (có dấu phân cách hàng nghìn), tính năng nổi bật và tình trạng hàng.
- Định dạng xác nhận ticket: Bắt buộc nêu rõ: Mã ticket, Tên khách hàng ghi nhận, Mức độ ưu tiên, Trạng thái xử lý và Cam kết thời gian phản hồi.

## 6. REASONING & RESPONSE PROTOCOL (Dành cho Agent Loop)
Thought: Phân tích kỹ nhu cầu người dùng, xác định rõ intent và quyết định tool/tham số cần gọi.
Action: {{"name": "<tên tool>", "args": {{<các tham số chuẩn xác>}}}}
Observation: <kết quả thực tế trả về từ hệ thống>
... (Tiếp tục suy luận nếu còn tool cần thực thi)
Final Answer: <câu trả lời hoàn chỉnh, trau chuốt, đầy đủ thông tin gửi khách hàng>
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop.
    Mục đích: So sánh chất lượng trả lời khi LLM bịa thông tin (hallucination).
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def query(self, user_input: str) -> Dict[str, Any]:
        """Gửi câu hỏi tới LLM (hoặc trả lời mock nếu không có API key)."""
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(
                    f"Bạn là chatbot tư vấn sản phẩm Vingroup. Trả lời câu hỏi sau "
                    f"mà KHÔNG có quyền tra cứu cơ sở dữ liệu hay gọi tool: {user_input}"
                )
                return {
                    "answer": response.text,
                    "tool_calls": [],
                    "status": "success",
                    "mode": "live_api"
                }
            except Exception:
                pass

        # Mock response — mô phỏng hallucination
        return {
            "answer": (
                "VinFast hiện có nhiều dòng xe điện với giá từ khoảng 300 triệu. "
                "Bạn có thể tham khảo tại showroom gần nhất hoặc website VinFast."
            ),
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Production-grade Agent với System Prompt Engineering & Tool Calling.

    Features:
      - 2 custom tools: search_product_catalog, submit_support_ticket
      - Sequential & Parallel tool calling
      - Max iterations safeguard
      - Full trace logging
    """

    def __init__(self, max_iterations: int = 5, api_key: str = None):
        self.max_iterations = max_iterations
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.trace: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------
    # Intent Detection (Rule-based Simulator)
    # -------------------------------------------------------------------

    def _detect_intents(self, user_input: str) -> Dict[str, Any]:
        """Phân tích câu hỏi người dùng để xác định tool cần gọi."""
        lower = user_input.lower()
        intents = {
            "needs_catalog": False,
            "needs_ticket": False,
            "is_faq": False,
            "catalog_args": {},
            "ticket_args": {}
        }

        # --- Detect FAQ intent FIRST (highest priority) ---
        faq_keywords = ["chính sách", "bảo hành", "đổi trả", "bao lâu", "điều kiện",
                        "warranty", "refund", "return"]
        action_keywords = ["xem", "tìm", "mua", "đặt", "giá dưới", "dưới"]
        ticket_action_keywords = ["phản hồi", "khiếu nại", "ghi nhận", "ticket", "phản ánh"]
        has_faq = any(kw in lower for kw in faq_keywords)
        has_action = any(kw in lower for kw in action_keywords)
        has_ticket_action = any(kw in lower for kw in ticket_action_keywords)

        if has_faq and not has_action and not has_ticket_action:
            intents["is_faq"] = True
            return intents

        # --- Detect catalog search intent ---
        catalog_keywords = ["xe điện", "xe vinfast", "vinfast", "resort", "vinpearl",
                            "du lịch", "sản phẩm", "xem", "tìm", "giá"]
        if any(kw in lower for kw in catalog_keywords) and has_action:
            # Determine category
            if any(kw in lower for kw in ["xe điện", "xe vinfast", "vinfast vf", "ô tô"]):
                intents["catalog_args"]["category"] = "xe_dien"
            elif any(kw in lower for kw in ["resort", "vinpearl", "du lịch", "nghỉ dưỡng", "khách sạn"]):
                intents["catalog_args"]["category"] = "du_lich"
            else:
                intents["catalog_args"]["category"] = "xe_dien"  # default

            # Parse max_price
            price_patterns = [
                (r"(\d+)\s*triệu", lambda m: int(m.group(1)) * 1_000_000),
                (r"(\d{3,})\s*000\s*000", lambda m: int(m.group(1)) * 1_000_000),
                (r"dưới\s*(\d+)\s*tr", lambda m: int(m.group(1)) * 1_000_000),
            ]
            for pattern, extractor in price_patterns:
                match = re.search(pattern, lower)
                if match:
                    intents["catalog_args"]["max_price"] = extractor(match)
                    break

            if intents["catalog_args"].get("category"):
                intents["needs_catalog"] = True

        # --- Detect ticket submit intent ---
        ticket_keywords = ["phản hồi", "khiếu nại", "lỗi", "hỏng", "bị", "hỗ trợ",
                           "ghi nhận", "ticket", "phản ánh", "sự cố"]
        if any(kw in lower for kw in ticket_keywords):
            # Extract customer name
            name_match = re.search(
                r"(?:tên\s+(?:tôi\s+là\s+|là\s+)?|tôi\s+là\s+|tên\s+)([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*)",
                user_input
            )
            if name_match:
                intents["ticket_args"]["customer_name"] = name_match.group(1).strip()
            else:
                intents["ticket_args"]["customer_name"] = "Khách hàng ẩn danh"

            # Extract issue description
            desc_patterns = [
                r"(?:xe|phòng|sản phẩm|dịch vụ).*?(?:bị|lỗi|hỏng|không hoạt động|sự cố)[^,.]*",
                r"(?:bị|lỗi|hỏng)[^,.]*",
            ]
            issue_desc = "Khách hàng cần hỗ trợ"
            for pattern in desc_patterns:
                desc_match = re.search(pattern, user_input, re.IGNORECASE)
                if desc_match:
                    issue_desc = desc_match.group(0).strip()
                    break
            intents["ticket_args"]["issue_description"] = issue_desc

            # Extract priority
            if any(kw in lower for kw in ["khẩn cấp", "nghiêm trọng", "gấp", "cấp bách"]):
                intents["ticket_args"]["priority"] = "high"
            elif any(kw in lower for kw in ["nhẹ", "không gấp", "khi nào cũng được"]):
                intents["ticket_args"]["priority"] = "low"
            else:
                intents["ticket_args"]["priority"] = "medium"

            if intents["ticket_args"].get("customer_name"):
                intents["needs_ticket"] = True

        return intents

    # -------------------------------------------------------------------
    # Step Execution Engine
    # -------------------------------------------------------------------

    def _execute_step(self, user_input: str, intents: Dict[str, Any], iteration: int) -> Tuple[str, bool]:
        """Thực thi từng bước suy luận của Agent."""

        # --- FAQ (No tool needed) ---
        if intents["is_faq"]:
            thought = "Đây là câu hỏi FAQ về chính sách bảo hành. Không cần gọi tool."
            final_answer = (
                "Chính sách bảo hành pin xe điện VinFast:\n"
                "- Pin xe VinFast được bảo hành 10 năm hoặc 200.000 km (tuỳ điều kiện nào đến trước).\n"
                "- Nếu dung lượng pin giảm dưới 70% trong thời gian bảo hành, VinFast sẽ thay thế miễn phí.\n"
                "- Chính sách bảo hành toàn quốc tại tất cả xưởng dịch vụ VinFast ủy quyền.\n"
                "- Hotline hỗ trợ: 1900-23-23-89."
            )
            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "final_answer": final_answer
            })
            return final_answer, True

        # --- Sequential Tool Calling ---
        if intents["needs_catalog"] and iteration == 1:
            args = intents["catalog_args"]
            max_price_display = f"{args['max_price']:,}" if 'max_price' in args else 'không giới hạn'
            thought = f"Khách hàng muốn tra cứu sản phẩm danh mục '{args.get('category', 'xe_dien')}' với giá tối đa {max_price_display} VNĐ."
            action = {"name": "search_product_catalog", "args": args}
            obs = TOOL_MAP["search_product_catalog"](**args)

            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "action": action,
                "observation": obs
            })

            if not intents["needs_ticket"]:
                # Single tool — generate final answer immediately
                if not obs or (isinstance(obs, list) and len(obs) == 0):
                    final_ans = f"Rất tiếc, hiện không tìm thấy sản phẩm nào thuộc danh mục '{args.get('category')}' với giá dưới {args.get('max_price', 0):,} VNĐ."
                elif isinstance(obs, list) and obs and "error" in obs[0]:
                    final_ans = f"Lỗi hệ thống: {obs[0]['error']}"
                else:
                    lines = []
                    for i, p in enumerate(obs, 1):
                        lines.append(f"{i}. **{p['name']}** — {p['price_vnd']:,} VNĐ")
                        lines.append(f"   {p['description']}")
                        lines.append(f"   Tình trạng: {'Còn hàng' if p['availability'] == 'in_stock' else 'Đặt trước'}")
                    final_ans = f"Tìm thấy {len(obs)} sản phẩm phù hợp:\n\n" + "\n".join(lines)
                return final_ans, True

            return f"Đã tra cứu catalog: {len(obs)} kết quả.", False

        if intents["needs_ticket"] and (iteration == 2 or (iteration == 1 and not intents["needs_catalog"])):
            args = intents["ticket_args"]
            thought = f"Cần ghi nhận ticket hỗ trợ cho khách hàng '{args['customer_name']}', mức ưu tiên: {args['priority']}."
            action = {"name": "submit_support_ticket", "args": args}
            obs = TOOL_MAP["submit_support_ticket"](**args)

            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "action": action,
                "observation": obs
            })

            if not intents["needs_catalog"]:
                # Single tool — generate final answer
                final_ans = (
                    f"Đã ghi nhận yêu cầu hỗ trợ của bạn:\n"
                    f"- **Mã ticket:** {obs['ticket_id']}\n"
                    f"- **Khách hàng:** {obs['customer_name']}\n"
                    f"- **Mức ưu tiên:** {obs['priority']}\n"
                    f"- **Trạng thái:** {obs['status']}\n\n"
                    f"Đội ngũ hỗ trợ Vingroup sẽ liên hệ bạn trong vòng 24 giờ."
                )
                return final_ans, True

            return f"Đã tạo ticket: {obs['ticket_id']}.", False

        # --- Final Answer Aggregation (iteration 3+, parallel case) ---
        thought = "Đã thu thập đủ thông tin từ cả 2 tool. Tổng hợp câu trả lời cho khách hàng."

        catalog_obs = next(
            (t["observation"] for t in self.trace
             if t.get("action", {}).get("name") == "search_product_catalog"),
            []
        )
        ticket_obs = next(
            (t["observation"] for t in self.trace
             if t.get("action", {}).get("name") == "submit_support_ticket"),
            {}
        )

        # Build catalog summary
        catalog_summary = "Không tìm thấy sản phẩm phù hợp."
        if catalog_obs and isinstance(catalog_obs, list) and len(catalog_obs) > 0:
            lines = []
            for i, p in enumerate(catalog_obs, 1):
                lines.append(f"   {i}. **{p['name']}** — {p['price_vnd']:,} VNĐ ({p['description'][:60]}...)")
            catalog_summary = "\n".join(lines)

        # Build ticket summary
        ticket_summary = "Không có ticket nào được tạo."
        if ticket_obs and isinstance(ticket_obs, dict) and "ticket_id" in ticket_obs:
            ticket_summary = (
                f"   - Mã ticket: {ticket_obs['ticket_id']}\n"
                f"   - Khách hàng: {ticket_obs['customer_name']}\n"
                f"   - Mức ưu tiên: {ticket_obs['priority']}\n"
                f"   - Trạng thái: {ticket_obs['status']}"
            )

        final_answer = (
            f"📋 **Kết quả tra cứu sản phẩm:**\n{catalog_summary}\n\n"
            f"🎫 **Thông tin ticket hỗ trợ:**\n{ticket_summary}\n\n"
            f"Cảm ơn bạn đã sử dụng dịch vụ Vingroup. Đội ngũ hỗ trợ sẽ liên hệ bạn sớm nhất!"
        )

        self.trace.append({
            "iteration": iteration,
            "thought": thought,
            "final_answer": final_answer
        })
        return final_answer, True

    # -------------------------------------------------------------------
    # Main Run Loop
    # -------------------------------------------------------------------

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop với safeguards."""
        self.trace = []
        intents = self._detect_intents(user_input)
        iteration = 1

        while iteration <= self.max_iterations:
            result, is_final = self._execute_step(user_input, intents, iteration)
            if is_final:
                return {
                    "answer": result,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed"
                }
            iteration += 1

        return {
            "answer": "Lỗi: Agent đã vượt quá số bước lặp tối đa (Max Iterations Safeguard).",
            "trace": self.trace,
            "iterations": iteration - 1,
            "status": "max_iterations_reached"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
