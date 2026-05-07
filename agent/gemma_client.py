import os
from typing import List, Dict, Any, Optional

# Attempt to load google-genai for Gemini proxy.
# In a real environment, this might connect to a local llama.cpp or HuggingFace server.
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

class GemmaClient:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.client = None
        if HAS_GENAI and self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            
    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Generates text from the LLM."""
        if not self.client:
            return self._mock_generate(prompt)
            
        try:
            config = types.GenerateContentConfig(
                temperature=0.2,
            )
            if system_instruction:
                config.system_instruction = system_instruction
                
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            return response.text
        except Exception as e:
            print(f"LLM API Error: {e}")
            return self._mock_generate(prompt)

    def _mock_generate(self, prompt: str) -> str:
        """Fallback mock generator for prototype demonstration without API keys."""
        prompt_lc = prompt.lower()
        
        if "incident" in prompt_lc or "action plan" in prompt_lc:
            return (
                "🚨 **แผนปฏิบัติการเร่งด่วน (Early Warning Action Plan)**\n\n"
                "1. **[ทันที]** เตรียมชาร์จ BESS โดยใช้พลังงานส่วนเกินจากสายส่งหลัก (Mainland) เพื่อสำรองไฟฟ้าสำหรับการเปลี่ยนถ่ายโหลด (Ref: SOP-BESS-2569-12 ส่วนที่ 4.1)\n"
                "2. **[+15 นาที]** เริ่มขั้นตอนการอุ่นเครื่อง (Warm-up) ของเครื่องปั่นไฟดีเซล DG-1 (Ref: SOP-DG-2569-04 ส่วนที่ 3.2)\n"
                "3. **[+30 นาที]** เชื่อมต่อ DG-1 เข้ากับระบบหากกำลังการผลิตจากโซลาร์เซลล์ลดลงต่ำกว่า 2 MW\n\n"
                "⚠️ **หากไม่ดำเนินการ:** คาดว่าจะเกิดไฟตกชั่วขณะ (Brownout) เป็นเวลา 5-10 นาทีในช่วงเปลี่ยนผ่านโหลดสูงสุด"
            )
        elif "forecast" in prompt_lc or "narrative" in prompt_lc:
            return (
                "📊 **วิเคราะห์แนวโน้มโหลด (Forecast Insight)**\n\n"
                "คาดการณ์ว่าในช่วง 24 ชั่วโมงข้างหน้า โหลดรวมจะพุ่งสูงขึ้นสูงสุดที่ **8.52 MW** ในเวลาประมาณ 19:30 น. "
                "เนื่องจากดัชนีความร้อน (Heat Index) ที่สูงถึง 39°C ประกอบกับช่วงเทศกาลสงกรานต์ที่มีนักท่องเที่ยวสะสมในพื้นที่\n\n"
                "💡 **คำแนะนำ:** แนะนำให้บริหารจัดการประจุไฟฟ้าใน BESS ให้พร้อมใช้งานก่อนเวลา 18:00 น. เพื่อลดการทำงานของเครื่องปั่นไฟสำรอง"
            )
        elif "schedule" in prompt_lc or "optimized" in prompt_lc:
            return (
                "💡 **การวิเคราะห์ความคุ้มค่า (Decision Explanation)**\n\n"
                "ตารางการสั่งจ่ายไฟฟ้าที่แนะนำ (MILP) จะช่วยประหยัดน้ำมันได้ประมาณ **145 ลิตรต่อวัน** เมื่อเทียบกับการปฏิบัติงานแบบปกติ\n"
                "- **ช่วง 14:00-16:00:** ใช้พลังงานจาก BESS แทนเครื่องปั่นไฟดีเซลในช่วงที่มีโหลดผันผวน\n"
                "- **ช่วง 19:00-21:00:** ปรับปรุงลำดับการทำงานของ DG Units ให้มีประสิทธิภาพสูงสุด (Optimal Load Factor)\n\n"
                "💰 **ผลประโยชน์รายปี:** ประหยัดน้ำมันได้ประมาณ 52,800 ลิตร คิดเป็นมูลค่าประมาณ **1.85 ล้านบาทต่อปี**"
            )
        
        return "สวัสดีครับ ผมคือ Gemma 4 AI Assistant ประจำระบบ GridTokenX มีอะไรให้ผมช่วยวิเคราะห์ข้อมูลโครงข่ายไฟฟ้าในวันนี้ไหมครับ?"

# Singleton instance
gemma_client = GemmaClient()
