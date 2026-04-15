import os
from gemini_integration import GeminiRefiner

def test_refinement_logic():
    # Mocking what happens in realtime.py
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
    refiner = GeminiRefiner(GEMINI_API_KEY)q4
    
    # Test Case 1: Initial glosses
    sentence_glosses = ["HELLO", "ME", "NAME", "CHANDRIKA"]
    print(f"Original Glosses: {sentence_glosses}")
    
    clean_glosses = [g for g in sentence_glosses if not g.startswith("Refined:")]
    if clean_glosses:
        final_sentence = refiner.refine_sentence(clean_glosses)
        print(f"Refined Sentence: {final_sentence}")
        sentence_glosses = [f"Refined: {final_sentence}"]
    
    print(f"Stored Sentence: {sentence_glosses}")
    
    # Test Case 2: Adding a new word after refinement
    sentence_glosses.append("SCHOOL")
    print(f"Glosses after adding 'SCHOOL': {sentence_glosses}")
    
    clean_glosses = [g for g in sentence_glosses if not g.startswith("Refined:")]
    print(f"Clean glosses for next refinement: {clean_glosses}")
    
    if clean_glosses:
        final_sentence = refiner.refine_sentence(clean_glosses)
        print(f"Final Refined Sentence: {final_sentence}")

if __name__ == "__main__":
    test_refinement_logic()
