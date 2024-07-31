import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env.

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

INITIAL_PROMPT = """Đóng vai bác sĩ phân loại ảnh ung thư tế bào học tuyến giáp.Đảm bảo rằng bạn sẽ dựa vào hệ thống phân loại 3 nhãn 0:B2-lành tính, 1:B5-nghi ngờ ác tính, 2:B6-ác tính.Thông tin có được tới thời điểm hiện tại là:"""
# 2.Nếu yêu cầu của người dùng không liên quan tới việc phân loại ung thư tuyến giáp 3 thang nêu trên hoặc hỏi không về cách sử dụng chương trình hỗ trợ bác sĩ này thì: 'Yêu cầu không liên quan đến chương trình hỗ trợ bác sĩ hỗ trợ phân loại ảnh ung thư tế bào học tuyến giáp. Vui lòng yêu cầu liên quan đến chủ đề hỗ trợ.'3
# +2.Không suy đoán và bịa đặt nội dung

class LLM_Model:
    def __init__(self, initial_prompt=""):
        if initial_prompt == "":
            initial_prompt = INITIAL_PROMPT
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        self.history = []
        self.initial_prompt = initial_prompt

    def generate_response(self, added_info, message):
        # Construct the prompt based on history and initial prompt
        prompt = (
            self.initial_prompt
            + added_info
            + 'Sau đây là lịch sử (tối đa 5 lần) trò chuyện trước đó: '
            + "\n".join(self.history[-5:])
            + f"\nUser: {message}"
        )  # Chỉ lấy 5 phần tử cuối của lịch sử

        # Generate response
        response = self.model.generate_content(prompt)

        print(f"Prompt: {prompt}")
        print(f"Response: {response.text}")

        # Append conversation to history
        self.history.append(f"User: {message}")
        self.history.append(f"AI: {response.text}")

        return response.text
