import google.generativeai as genai
import os

class GeminiRefiner:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
    def refine_sentence(self, gloss_sequence):
        """
        Converts a list of ASL glosses into a grammatically correct English sentence.
        Example: ['HELLO', 'ME', 'NAME', 'CHANDRIKA'] -> 'Hello, my name is Chandrika.'
        """
        if not gloss_sequence:
            return ""
            
        prompt = f"""
        Context: You are an expert ASL (American Sign Language) interpreter.
        Task: Convert the following sequence of predicted ASL glosses into a natural, grammatically correct English sentence.
        
        Glosses: {', '.join(gloss_sequence)}
        
        Rules:
        1. Fix the grammar and tense.
        2. Remove redundant or repeated words.
        3. Keep the meaning intact.
        4. Output ONLY the refined sentence.
        
        Refined Sentence:
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Gemini API Error: {e}")
            return " ".join(gloss_sequence) # Fallback to raw glosses

if __name__ == "__main__":
    # Test with placeholder key
    API_KEY = "YOUR_GEMINI_API_KEY"
    refiner = GeminiRefiner(API_KEY)
    test_glosses = ["HELLO", "ME", "GO", "SCHOOL", "TODAY"]
    print(f"Original: {test_glosses}")
    # print(f"Refined: {refiner.refine_sentence(test_glosses)}")
