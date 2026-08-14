"""
Rule Engine Architecture implementing:
1. Baseline 0b Keyword Rules for Disaster Classification
2. Approach A Extraction Rules (Regex for Phone, Coordinates, Map URL, and Counts)
3. Pediatric IITT Rule Engine (Task 3.2) & Adult IITT Rule Engine (Task 3.3)
"""

import re
from typing import Dict, Any, List, Optional, Tuple


class SimpleKeywordRules:
    """Baseline 0b Rule-Based Disaster Keyword Classifier."""
    
    HELP_KEYWORDS = [
        r"ช่วยด้วย", r"ขอความช่วยเหลือ", r"ติดค้าง", r"ขอเรือ", r"ถุงยังชีพ",
        r"ขาดอาหาร", r"น้ำท่วมสูง", r"ผู้ป่วยติดเตียง", r"อพยพ", r"เสียชีวิต",
        r"เด็กเล็ก", r"คนแก่", r"จมน้ำ", r"ต้องการอาหาร", r"ต้องการน้ำ"
    ]
    
    def __init__(self):
        self.pattern = re.compile("|".join(self.HELP_KEYWORDS), re.IGNORECASE)
        
    def predict_one(self, text: str) -> int:
        if self.pattern.search(str(text)):
            return 1
        return 0
        
    def predict(self, texts: List[str]) -> List[int]:
        return [self.predict_one(t) for t in texts]


class ExtractionRulesEngine:
    """Approach A Rule-Based Extraction Engine for Phone, Coordinates, Map URLs, Locations, and Counts."""
    
    PHONE_REGEX = re.compile(r"(?:0[689][0-9][\s-]?[0-9]{3,4}[\s-]?[0-9]{4}|0[0-9]{1,2}[\s-]?[0-9]{3}[\s-]?[0-9]{4})")
    COORDS_REGEX = re.compile(r"(\d{1,2}\.\d{3,})\s*,\s*(9[7-9]\.\d{3,}|10[0-5]\.\d{3,})")
    MAP_URL_REGEX = re.compile(r"https?://(?:maps\.app\.goo\.gl|goo\.gl/maps|www\.google\.com/maps|maps\.google\.com)[^\s,]+")
    try:
        from pythainlp.corpus import provinces
        _PROV_LIST = "|".join(sorted(list(provinces()), key=len, reverse=True))
    except Exception:
        _PROV_LIST = "กรุงเทพมหานคร|กรุงเทพฯ|เชียงใหม่|เชียงราย|พะเยา|กาญจนบุรี|สมุทรสาคร|นครนายก|ชลบุรี|พิษณุโลก|ภูเก็ต|สงขลา|น่าน|แพร่|ลำปาง|ลำพูน|แม่ฮ่องสอน|ตาก|สุโขทัย|อุตรดิตถ์|กำแพงเพชร|พิจิตร|เพชรบูรณ์|นครสวรรค์|อุทัยธานี|นนทบุรี|ปทุมธานี|พระนครศรีอยุธยา|อ่างทอง|ลพบุรี|สิงห์บุรี|ชัยนาท|สระบุรี|นครปฐม|สุพรรณบุรี|สมุทรปราการ|สมุทรสงคราม|เพชรบุรี|ประจวบคีรีขันธ์|ราชบุรี|ฉะเชิงเทรา|ปราจีนบุรี|สระแก้ว|ระยอง|จันทบุรี|ตราด|นครราชสีมา|บุรีรัมย์|สุรินทร์|ศรีสะเกษ|อุบลราชธานี|ยโสธร|ชัยภูมิ|อำนาจเจริญ|บึงกาฬ|หนองบัวลำภู|ขอนแก่น|อุดรธานี|เลย|หนองคาย|มหาสารคาม|ร้อยเอ็ด|กาฬสินธุ์|สกลนคร|นครพนม|มุกดาหาร|นครศรีธรรมราช|กระบี่|พังงา|สุราษฎร์ธานี|ระนอง|ชุมพร|ตรัง|พัทลุง|สตูล|ปัตตานี|ยะลา|นราธิวาส"

    LOCATION_PATTERNS = [
        # Full hierarchy: Landmark + Sub-district + District + Province
        re.compile(
            r"((?:(?:ตลาดน้ำ|ตลาด|วัด|ซอย|ถนน|หมู่บ้าน|หมู่ที่|ชุมชน|คอนโด|อพาร์ทเม้นท์|โรงเรียน|รพ\.|สะพาน|บ้าน|พื้นที่)\s*[ก-๙0-9\.\-\/\(\)\s]+?)?"
            r"(?:ตำบล|ต\.|แขวง)\s*[ก-๙0-9]+"
            r"(?:\s*(?:อำเภอ|อ\.|เขต)\s*[ก-๙0-9]+)?"
            r"(?:\s*(?:จังหวัด|จ\.)?\s*(?:" + _PROV_LIST + r"))?)"
        ),
        # District + Province
        re.compile(
            r"((?:(?:ตลาดน้ำ|ตลาด|วัด|ซอย|ถนน|หมู่บ้าน|หมู่ที่|ชุมชน|คอนโด|อพาร์ทเม้นท์|โรงเรียน|รพ\.|สะพาน|บ้าน|พื้นที่)\s*[ก-๙0-9\.\-\/\(\)\s]+?)?"
            r"(?:อำเภอ|อ\.|เขต)\s*[ก-๙0-9]+"
            r"(?:\s*(?:จังหวัด|จ\.)?\s*(?:" + _PROV_LIST + r")))"
        ),
        # Landmark with Province
        re.compile(
            r"((?:(?:ตลาดน้ำ|ตลาด|วัด|ซอย|ถนน|หมู่บ้าน|หมู่ที่|ชุมชน|คอนโด|อพาร์ทเม้นท์|โรงเรียน|รพ\.|สะพาน|บ้าน|พื้นที่)\s*[ก-๙0-9\.\-\/\(\)\s]+?)"
            r"(?:จังหวัด|จ\.)\s*(?:" + _PROV_LIST + r"))"
        )
    ]
    
    # Count patterns
    COUNT_PATTERNS = {
        "gt_dead": [r"เสียชีวิต\s*(\d+)", r"ตาย\s*(\d+)", r"ศพ\s*(\d+)"],
        "gt_critical": [r"ผู้ป่วยหนัก\s*(\d+)", r"สาหัส\s*(\d+)", r"วิกฤต\s*(\d+)"],
        "gt_urgent": [r"ผู้ป่วยฉุกเฉิน\s*(\d+)", r"บาดเจ็บ\s*(\d+)"],
        "gt_safe": [r"ปลอดภัย\s*(\d+)", r"รอด\s*(\d+)"],
        "gt_child": [r"เด็ก\s*(\d+)", r"ทารก\s*(\d+)"],
        "gt_bedridden": [r"ผู้ป่วยติดเตียง\s*(\d+)", r"ติดเตียง\s*(\d+)"],
        "gt_item_firstaid": [r"ชุดยาสมุนไพร\s*(\d+)", r"ยา\s*(\d+)", r"ชุดปฐมพยาบาล\s*(\d+)"],
        "gt_item_food": [r"ข้าว\s*(\d+)", r"อาหาร\s*(\d+)", r"กล่อง\s*(\d+)", r"ชุดอาหาร\s*(\d+)"],
        "gt_item_energy": [r"แบตเตอรี่\s*(\d+)", r"เทียน\s*(\d+)", r"ไฟฉาย\s*(\d+)"]
    }
    
    def extract_phone(self, text: str) -> Optional[str]:
        m = self.PHONE_REGEX.search(str(text))
        return m.group(0).strip() if m else None

    def extract_location(self, text: str) -> Optional[str]:
        cleaned_text = str(text or "")
        for pattern in self.LOCATION_PATTERNS:
            m = pattern.search(cleaned_text)
            if m:
                loc = m.group(0).strip()
                # Clean trailing noise/polite particles
                loc = re.sub(r"[\r\n\t]+", " ", loc)
                loc = re.sub(r"\s*(?:ครับ|ค่ะ|คะ|นะ|จ้า|ด่วน|เลย|นะคร้าบ|นะค่ะ|นะคะ|ตอนนี้|พิกัด).*$", "", loc)
                if len(loc.strip()) > 3:
                    return loc.strip()
        return None
        
    def extract_coords(self, text: str) -> Tuple[Optional[float], Optional[float]]:
        m = self.COORDS_REGEX.search(str(text))
        if m:
            try:
                return float(m.group(1)), float(m.group(2))
            except ValueError:
                pass
        return None, None
        
    def extract_map_url(self, text: str) -> Optional[str]:
        m = self.MAP_URL_REGEX.search(str(text))
        return m.group(0).strip() if m else None
        
    def extract_count(self, text: str, field_name: str) -> int:
        patterns = self.COUNT_PATTERNS.get(field_name, [])
        for pat in patterns:
            m = re.search(pat, str(text))
            if m:
                try:
                    return int(m.group(1))
                except ValueError:
                    pass
        return 0
        
    def extract_all_counts(self, text: str) -> Dict[str, int]:
        return {field: self.extract_count(text, field) for field in self.COUNT_PATTERNS.keys()}


class ClauseSplitterRules:
    """Method 3.1a Rule-based Clause Splitter for People & Symptoms Extraction."""
    
    CONJUNCTION_PATTERN = re.compile(r"(?:\n|และ|กับ|ส่วน|รวมถึง|\s{2,}|,)")
    AGE_PATTERN = re.compile(r"(?:อายุ|เด็ก)\s*(\d{1,2})\s*(?:ปี|ขวบ)?")
    CHILD_KEYWORDS = [r"เด็ก", r"ทารก", r"ขวบ", r"หลาน", r"ลูกเล็ก"]
    NAME_PATTERN = re.compile(r"(?:คุณ|นาย|นาง|นางสาว|เด็กชาย|เด็กหญิง|น้อง|พี่)\s*([ก-๙]+)")
    
    def split_clauses(self, text: str) -> List[str]:
        clauses = self.CONJUNCTION_PATTERN.split(str(text))
        return [c.strip() for c in clauses if len(c.strip()) > 3]
        
    def extract_clauses(self, text: str) -> List[str]:
        return self.split_clauses(text)
        
    def extract_victims(self, text: str) -> List[Dict[str, Any]]:
        clauses = self.split_clauses(text)
        victims = []
        
        for clause in clauses:
            m_age = self.AGE_PATTERN.search(clause)
            age = int(m_age.group(1)) if m_age else None
            
            is_child = False
            if age is not None and age <= 12:
                is_child = True
            else:
                for kw in self.CHILD_KEYWORDS:
                    if re.search(kw, clause):
                        is_child = True
                        break
                        
            m_name = self.NAME_PATTERN.search(clause)
            name = m_name.group(1) if m_name else None
            
            victims.append({
                "name": name,
                "age": age,
                "age_group": "child" if is_child else "adult",
                "is_child": is_child,
                "symptoms_literal": clause
            })
            
        if not victims:
            victims = [{
                "name": None,
                "age": None,
                "age_group": "adult",
                "is_child": False,
                "symptoms_literal": str(text)
            }]
            
        return victims


class PediatricIITTRules:
    """Pediatric Clinical Triage Rule Engine (Child <= 12 years)."""
    
    RED_PATTERNS = [r"หมดสติ", r"ไม่รู้สึกตัว", r"หยุดหายใจ", r"ตัวเขียว", r"ชัก", r"ช็อค", r"เลือดไหลไม่หยุด", r"หอบหนักมาก"]
    YELLOW_PATTERNS = [r"กระดูกหัก", r"แขนหัก", r"ขาหัก", r"ไข้สูง", r"ซึม", r"ปวดท้องรุนแรง", r"แผลใหญ่"]
    
    def predict_one(self, text: str) -> str:
        s = str(text)
        for pat in self.RED_PATTERNS:
            if re.search(pat, s):
                return "RED"
        for pat in self.YELLOW_PATTERNS:
            if re.search(pat, s):
                return "YELLOW"
        return "GREEN"
        
    def classify(self, text: str) -> str:
        return self.predict_one(text)
        
    def predict(self, texts: List[str]) -> List[str]:
        return [self.predict_one(t) for t in texts]


class AdultIITTRules:
    """Adult Clinical Triage Rule Engine (Age > 12 years)."""
    
    RED_PATTERNS = [r"หมดสติ", r"ไม่รู้สึกตัว", r"หยุดหายใจ", r"แน่นหน้าอก", r"ช็อค", r"เสียเลือดมาก", r"ติดเตียงอาการหนัก", r"วิกฤต"]
    YELLOW_PATTERNS = [r"กระดูกหัก", r"ขาหัก", r"แขนหัก", r"แผลฉีกขาด", r"ปวดท้อง", r"ไข้สูง", r"บาดเจ็บ"]
    
    def predict_one(self, text: str) -> str:
        s = str(text)
        for pat in self.RED_PATTERNS:
            if re.search(pat, s):
                return "RED"
        for pat in self.YELLOW_PATTERNS:
            if re.search(pat, s):
                return "YELLOW"
        return "GREEN"
        
    def classify(self, text: str) -> str:
        return self.predict_one(text)
        
    def predict(self, texts: List[str]) -> List[str]:
        return [self.predict_one(t) for t in texts]

