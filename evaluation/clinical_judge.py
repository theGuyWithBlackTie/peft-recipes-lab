import json
import google.generativeai as genai
from typing import Dict, Any

def evaluate_clinical_reasoning(question: str, reference_answer: str, model_generation: str, model_name: str = "gemini-2.5-pro") -> Dict[str, int]:
    """
    Evaluates clinical reasoning using Gemini API along 4 specific axes.
    """
    prompt = f"""
You are an expert clinical reasoning judge. Evaluate the model's generated response to the medical case below.

Medical Case / Question:
{question}

Reference Correct Answer:
{reference_answer}

Model Generation (including <think> tags if any):
{model_generation}

Evaluate the model's reasoning process along the following 4 axes, scoring each from 1 to 5 (1 = poor, 5 = excellent):
1. intermediate_hypotheses: Did the model form clear intermediate clinical hypotheses rather than jumping straight to conclusion?
2. evidence_utilization: How thoroughly and accurately does the model extract and connect patient symptoms, history, and physical findings?
3. contradiction_handling: How effectively does the model identify conflicting signs, resolve ambiguity, and rule out competing conditions?
4. reflection_correction: Does the model actively audit its own line of reasoning, question early assumptions, or double-check its deductions?

Respond EXACTLY with a JSON object in this format:
{{
  "intermediate_hypotheses": <int>,
  "evidence_utilization": <int>,
  "contradiction_handling": <int>,
  "reflection_correction": <int>
}}
"""
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        result = json.loads(response.text)
        
        axes = [
            "intermediate_hypotheses",
            "evidence_utilization",
            "contradiction_handling",
            "reflection_correction"
        ]
        
        for axis in axes:
            if axis not in result:
                result[axis] = 1
            else:
                try:
                    result[axis] = int(result[axis])
                except ValueError:
                    result[axis] = 1
                    
        return result
    except Exception as e:
        print(f"Error evaluating clinical reasoning: {e}")
        return {
            "intermediate_hypotheses": 1,
            "evidence_utilization": 1,
            "contradiction_handling": 1,
            "reflection_correction": 1
        }
