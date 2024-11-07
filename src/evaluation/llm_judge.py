"""
This script demonstrates the pattern of using an LLM to evaluate another LLM's response.
"""

def judge_response(query, reference, student_answer):
    """
    Mocking an LLM call that judges the student_answer based on reference.
    """
    # In a real scenario, this would be a prompt to GPT-4/Claude
    print(f"DEBUG: Judging Response...")
    print(f"QUERY: {query}")
    print(f"REFERENCE: {reference}")
    print(f"STUDENT: {student_answer}")
    
    # Simple logic to simulate judgment
    if len(student_answer) < 5:
        return {"score": 1, "reason": "Answer too short."}
    if reference.lower() in student_answer.lower():
        return {"score": 10, "reason": "Accurate and grounded."}
    
    return {"score": 5, "reason": "Partially correct but lacks detail."}

if __name__ == "__main__":
    query = "What is backpropagation?"
    reference = "An algorithm used for training neural networks by calculating gradients."
    
    student_a = "It is how NNs learn by updating weights using gradients."
    student_b = "I don't know."
    
    print("--- Result A ---")
    print(judge_response(query, reference, student_a))
    
    print("\n--- Result B ---")
    print(judge_response(query, reference, student_b))
