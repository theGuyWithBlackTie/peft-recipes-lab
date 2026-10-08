import re
from typing import Dict, Any, List

def parse_cot_response(text: str) -> Dict[str, Any]:
    """
    Parses a Chain-of-Thought response.
    Checks for <think> and </think> tags, extracts thought and final answer.
    """
    result = {
        "thought": "",
        "answer": "",
        "thought_tokens": 0,
        "answer_tokens": 0,
        "total_tokens": 0,
        "reasoning_ratio": 0.0,
        "reflection_markers_count": 0,
        "reflection_marker_density": 0.0,
        "format_valid": False,
        "unclosed_tags": False,
        "malformed_tags": False
    }

    def count_tokens(s):
        return len(s.split())

    total_tokens = count_tokens(text)
    result["total_tokens"] = total_tokens

    has_think_start = "<think>" in text
    has_think_end = "</think>" in text

    if not has_think_start and not has_think_end:
        result["answer"] = text.strip()
        result["answer_tokens"] = count_tokens(result["answer"])
        return result

    if has_think_start and not has_think_end:
        result["unclosed_tags"] = True
        result["malformed_tags"] = True
        parts = text.split("<think>")
        if len(parts) > 1:
            result["thought"] = parts[1].strip()
            result["thought_tokens"] = count_tokens(result["thought"])
        if total_tokens > 0:
            result["reasoning_ratio"] = result["thought_tokens"] / total_tokens
        return result

    if not has_think_start and has_think_end:
        result["malformed_tags"] = True
        parts = text.split("</think>")
        if len(parts) > 1:
            result["answer"] = parts[1].strip()
            result["answer_tokens"] = count_tokens(result["answer"])
        return result

    # Standard case
    thought_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    match = thought_pattern.search(text)
    
    if match:
        result["format_valid"] = True
        thought = match.group(1).strip()
        result["thought"] = thought
        
        answer = text[match.end():].strip()
        result["answer"] = answer
        
        result["thought_tokens"] = count_tokens(thought)
        result["answer_tokens"] = count_tokens(answer)
        
        if total_tokens > 0:
            result["reasoning_ratio"] = result["thought_tokens"] / total_tokens

        cognitive_words = [
            r'\bwait\b', r'\blet me check\b', r'\brecheck\b', r'\bhowever\b',
            r'\balternatively\b', r'\bcould it be\b', r'\brule out\b',
            r'\bunlikely\b', r'\bmight be wrong\b'
        ]
        
        marker_count = 0
        thought_lower = thought.lower()
        for word in cognitive_words:
            marker_count += len(re.findall(word, thought_lower))
        
        result["reflection_markers_count"] = marker_count
        if result["thought_tokens"] > 0:
            result["reflection_marker_density"] = marker_count / result["thought_tokens"]
    else:
        result["malformed_tags"] = True

    return result

def compute_structural_metrics(generations: List[str]) -> Dict[str, float]:
    """
    Computes structural metrics across a batch of generations.
    """
    if not generations:
        return {}

    total_generations = len(generations)
    valid_format_count = 0
    total_thought_tokens = 0
    total_answer_tokens = 0
    total_reasoning_ratio = 0.0
    total_reflection_markers = 0
    
    for gen in generations:
        parsed = parse_cot_response(gen)
        
        if parsed["format_valid"]:
            valid_format_count += 1
            
        total_thought_tokens += parsed["thought_tokens"]
        total_answer_tokens += parsed["answer_tokens"]
        total_reasoning_ratio += parsed["reasoning_ratio"]
        total_reflection_markers += parsed["reflection_markers_count"]
        
    return {
        "format_compliance_rate": valid_format_count / total_generations,
        "avg_thought_tokens": total_thought_tokens / total_generations,
        "avg_answer_tokens": total_answer_tokens / total_generations,
        "avg_reasoning_ratio": total_reasoning_ratio / total_generations,
        "avg_reflection_markers": total_reflection_markers / total_generations
    }
