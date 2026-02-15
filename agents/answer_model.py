#!/usr/bin/env python3
"""
AAIPL A-Agent: Answer Model (OPTIMIZED FOR SPEED)
Generates answers for logical reasoning questions across multiple topics:
- Syllogisms
- Seating Arrangements (Circular and Linear)
- Blood Relations and Family Tree
- Mixed Series (Alphanumeric)

Requirements:
- Time limit: 9 seconds per answer, 900 seconds for 100 answers
- Reasoning limit: 100 words maximum
- No hardcoding allowed
- Must follow exact JSON format

OPTIMIZATION FOCUS: Speed over token counting
"""

import json
import argparse
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class AAgent:
    """A-Agent for generating answers to logical reasoning questions."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        device: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = True,
        max_new_tokens: int = 70,  # Reduced from 80 for faster generation
        temperature: float = 0.3,  # Reduced for more deterministic, faster responses
        top_p: float = 0.9,
        cache_dir: Optional[str] = None,
        time_limit_per_answer: float = 9.0,  # Hard time limit per answer
        **kwargs
    ):
        """
        Initialize the Answer Model.
        
        Args:
            model_name: HuggingFace model identifier (default: 'Qwen/Qwen3-4B')
            device: Device to run model on ('cuda', 'cpu', or 'auto')
            load_in_4bit: Whether to load model in 4-bit quantization
            load_in_8bit: Whether to load model in 8-bit quantization (default: True)
            max_new_tokens: Maximum tokens to generate (default: 70, optimized for speed)
            temperature: Sampling temperature (default: 0.3, lower for faster generation)
            top_p: Nucleus sampling parameter (default: 0.9)
            cache_dir: Directory to cache downloaded models (default: ./model_cache)
            time_limit_per_answer: Time limit per answer in seconds (default: 9.0)
            **kwargs: Additional arguments (ignored, for framework compatibility)
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.time_limit_per_answer = time_limit_per_answer
        
        # Set cache directory - use workspace if not specified
        if cache_dir is None:
            cache_dir = "./model_cache"
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"⚡ Loading model: {model_name} (Speed-optimized)")
        print(f"📁 Cache directory: {cache_dir}")
        print(f"⏱️  Time limit: {time_limit_per_answer}s per answer")
        print(f"🔢 Max generation: {max_new_tokens} tokens")
        
        # Configure quantization if requested
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
            cache_dir=cache_dir
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with speed optimizations
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
            "cache_dir": cache_dir,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        else:
            if device == "auto":
                model_kwargs["device_map"] = "auto"
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Move to device if not using device_map
        if device != "auto" and quantization_config is None:
            self.model = self.model.to(device)
        
        self.model.eval()
        print(f"✅ Model loaded successfully on {self.model.device}")
        
        # Print GPU info if available
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("💻 Running on CPU")
    
    def create_prompt(self, question_data: Dict[str, Any]) -> str:
        """
        Create a concise prompt for answering a question.
        Optimized for speed - shorter prompts generate faster.
        
        Args:
            question_data: Dictionary containing 'topic', 'question', 'choices', etc.
            
        Returns:
            Formatted prompt string
        """
        topic = question_data.get("topic", "")
        question = question_data.get("question", "")
        choices = question_data.get("choices", [])
        
        # Format choices with letters A, B, C, D
        choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        
        # Concise prompt optimized for fast, accurate responses
        prompt = f"""Solve this {topic} question.

{question}

{choices_text}

Respond ONLY in this format:
Answer: [A/B/C/D]
Reason: [Brief explanation in 1-2 sentences]"""
        
        return prompt
    
    def parse_response(self, response: str, choices: List[str]) -> tuple[str, str]:
        """
        Parse the model's response to extract answer and reasoning.
        EMERGENCY FIX: More robust parsing with debug output
        
        Args:
            response: Raw model output
            choices: List of answer choices
            
        Returns:
            Tuple of (answer, reasoning)
        """
        # Clean the response
        response = response.strip()
        
        # EMERGENCY DEBUG: Print raw response for troubleshooting
        if len(response) < 10:
            print(f"⚠️  WARNING: Very short response ({len(response)} chars): '{response}'")
        
        # Try to find the answer choice
        answer = None
        reasoning = response
        
        # Look for patterns like "Answer: A", "The answer is B", etc.
        import re
        
        # Strategy 1: Look for "Answer: X" format (most reliable)
        answer_match = re.search(r'Answer:\s*([A-Da-d])', response, re.IGNORECASE)
        if answer_match:
            letter = answer_match.group(1).upper()
            letter_index = ord(letter) - ord('A')
            if 0 <= letter_index < len(choices):
                answer = choices[letter_index]
                
                # Try to extract reason/reasoning
                reason_match = re.search(r'(?:Reason|Explanation):\s*(.+)', response, re.IGNORECASE | re.DOTALL)
                if reason_match:
                    reasoning = reason_match.group(1).strip()
        
        # Strategy 2: Look for letter patterns at start of response
        if answer is None:
            patterns = [
                r'^\s*([A-Da-d])[\.\):\s]',  # A. or A) or A: at start
                r'\(([A-Da-d])\)',            # (A)
                r'(?:answer|choice|option|correct)(?:\s+is)?(?:\s*:)?\s*([A-Da-d])',
                r'^([A-Da-d])\s*[-–—]\s*',   # A - ...
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
                if match:
                    letter = match.group(1).upper()
                    letter_index = ord(letter) - ord('A')
                    if 0 <= letter_index < len(choices):
                        answer = choices[letter_index]
                        break
        
        # Strategy 3: Look for number patterns (1-4) as fallback
        if answer is None:
            number_patterns = [
                r'(?:answer|choice|option)(?:\s+is)?(?:\s*:)?\s*(\d)',
                r'^\s*(\d)[\.\):]',
            ]
            
            for pattern in number_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    answer_num = int(match.group(1))
                    if 1 <= answer_num <= len(choices):
                        answer = choices[answer_num - 1]
                        break
        
        # Strategy 4: Check if any choice text appears in response
        if answer is None:
            for choice in choices:
                # Match significant words from choice
                choice_words = set(w.lower() for w in choice.split() if len(w) > 3)
                response_words = set(w.lower() for w in response.split())
                
                if choice_words:
                    overlap = len(choice_words & response_words)
                    if overlap >= max(1, len(choice_words) * 0.6):
                        answer = choice
                        break
        
        # EMERGENCY: If response is just a single character or very short, it's likely corrupt
        # Return first choice as safe fallback
        if answer is None:
            if len(response) < 5 or not any(c.isalpha() for c in response):
                print(f"⚠️  EMERGENCY FALLBACK: Response too short or corrupt, using first choice")
                answer = choices[0] if choices else "Error"
                reasoning = "Response parsing failed - used fallback answer"
            elif choices:
                answer = choices[0]
        
        # Truncate reasoning to meet 100-word limit
        words = reasoning.split()
        if len(words) > 100:
            reasoning = ' '.join(words[:100]) + "..."
        
        return answer, reasoning
    
    def count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
    def answer(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Alias for generate_answer for framework compatibility.
        
        Args:
            question_data: Dictionary containing question information
            
        Returns:
            Dictionary with 'answer' and 'reasoning' keys
        """
        return self.generate_answer(question_data)
    
    def generate_response(self, prompt: str, system_prompt: str = "", **kwargs) -> tuple:
        """
        Generate response from a text prompt (framework compatibility method).
        
        This method is called by the hackathon framework.
        
        Args:
            prompt: The prompt text to generate response for
            system_prompt: System prompt (not used, for compatibility)
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (response_text, time_limit, generation_time)
        """
        start_time = time.time()
        
        try:
            # Tokenize the prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1536
            )
            
            # Move to device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate with optimizations for speed
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    num_beams=1,  # Beam search disabled for speed
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            generation_time = time.time() - start_time
            
            return response, self.time_limit_per_answer, generation_time
            
        except Exception as e:
            print(f"❌ Error in generate_response: {e}")
            generation_time = time.time() - start_time
            return f"Error: {str(e)}", self.time_limit_per_answer, generation_time
    
    def generate_answer(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an answer for a given question.
        Optimized for speed - focuses on meeting time limit.
        EMERGENCY FIX: Added debug output
        
        Args:
            question_data: Dictionary containing question information
            
        Returns:
            Dictionary with 'answer' and 'reasoning' keys
        """
        start_time = time.time()
        
        try:
            # Create prompt
            prompt = self.create_prompt(question_data)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1536
            )
            
            # Move to device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate with speed optimizations
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    num_beams=1,  # Single beam for speed
                )
            
            # Decode
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # EMERGENCY DEBUG: Print raw response for first few questions
            print(f"\n🔍 RAW RESPONSE (length: {len(response)}):")
            print(f"'{response[:200]}'")  # First 200 chars
            
            # Parse response
            choices = question_data.get("choices", [])
            answer, reasoning = self.parse_response(response, choices)
            
            # EMERGENCY DEBUG: Print extracted answer
            print(f"📝 EXTRACTED ANSWER: '{answer}'")
            print(f"📝 REASONING: '{reasoning[:100]}'")
            
            # Clean up GPU memory if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            elapsed_time = time.time() - start_time
            
            # EMERGENCY: Make absolutely sure return values are strings
            final_answer = str(answer) if answer is not None else (choices[0] if choices else "Error")
            final_reasoning = str(reasoning) if reasoning is not None else "No reasoning generated"
            
            # EMERGENCY: Verify answer is not empty or just whitespace
            if not final_answer or not final_answer.strip():
                print(f"⚠️  EMPTY ANSWER DETECTED, using first choice")
                final_answer = choices[0] if choices else "Error"
            
            result = {
                "answer": final_answer,
                "reasoning": final_reasoning
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            import traceback
            traceback.print_exc()
            choices = question_data.get("choices", [])
            return {
                "answer": choices[0] if choices else "Error",
                "reasoning": f"Error: {str(e)}"
            }
    
    def process_questions(
        self, 
        questions: List[Dict[str, Any]], 
        output_file: Optional[str] = None,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process a list of questions and generate answers.
        Tracks timing to ensure 9-second and 900-second limits are met.
        
        Args:
            questions: List of question dictionaries
            output_file: Optional path to save answers
            verbose: Whether to print progress
            
        Returns:
            List of answer dictionaries
        """
        answers = []
        total_time = 0
        time_violations = 0
        
        print("\n" + "="*60)
        print(f"⚡ Processing {len(questions)} questions (Speed-optimized)")
        print("="*60 + "\n")
        
        for i, question_data in enumerate(questions, 1):
            start_time = time.time()
            
            if verbose:
                print(f"\n=== Question {i}/{len(questions)} ===")
                print(f"Topic: {question_data.get('topic', 'N/A')}")
                print(f"Question: {question_data.get('question', '')[:80]}...")
            
            # Generate answer
            result = self.generate_answer(question_data)
            
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            
            # Check time limit
            if elapsed_time > self.time_limit_per_answer:
                time_violations += 1
                print(f"⚠️  TIME VIOLATION: {elapsed_time:.2f}s > {self.time_limit_per_answer}s")
            
            if verbose:
                print(f"Answer: {result['answer']}")
                print(f"Time: {elapsed_time:.2f}s")
            
            answers.append(result)
            
            # Progress indicator every 10 questions
            if i % 10 == 0:
                avg_time = total_time / i
                estimated_remaining = avg_time * (len(questions) - i)
                estimated_total = total_time + estimated_remaining
                
                print(f"\n📊 Progress: {i}/{len(questions)}")
                print(f"   Average time: {avg_time:.2f}s per question")
                print(f"   Estimated total: {estimated_total:.0f}s")
                
                if estimated_total > 900:
                    print(f"   ⚠️  WARNING: Projected to exceed 900s limit!")
        
        # Final summary
        print("\n" + "="*60)
        print("✅ PROCESSING COMPLETE")
        print("="*60)
        print(f"📊 Total questions: {len(questions)}")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"⚡ Average time: {total_time/len(questions):.2f}s per question")
        print(f"⚠️  Time violations: {time_violations} questions over {self.time_limit_per_answer}s")
        
        # Check total time constraint
        if total_time > 900:
            print(f"\n🚨 FAILED: Exceeded 900s (15 minutes) total time limit!")
            print(f"   Exceeded by: {total_time - 900:.0f}s")
        else:
            print(f"\n✅ PASSED: Within 900s time limit!")
            print(f"   Time remaining: {900 - total_time:.0f}s")
        
        # Check per-answer time constraint
        if time_violations == 0:
            print(f"✅ PASSED: All answers under {self.time_limit_per_answer}s!")
        else:
            print(f"⚠️  WARNING: {time_violations} answers exceeded {self.time_limit_per_answer}s limit")
        
        # Save to file if specified
        if output_file:
            self.save_answers(answers, output_file)
        
        return answers
    
    def save_answers(self, answers: List[Dict[str, Any]], output_file: str):
        """Save answers to a JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(answers, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Answers saved to: {output_file}")


def load_questions(input_file: str) -> List[Dict[str, Any]]:
    """Load questions from a JSON file."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both list format and dict with 'questions' key
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'questions' in data:
        return data['questions']
    else:
        raise ValueError("Invalid input format. Expected list or dict with 'questions' key")


def main():
    """Main entry point for the answer model."""
    parser = argparse.ArgumentParser(
        description="AAIPL A-Agent: Generate answers (Speed-optimized)"
    )
    
    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-4B",
        help="HuggingFace model name (default: 'Qwen/Qwen3-4B')"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run model on"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit quantization"
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8-bit quantization (recommended for speed)"
    )
    
    # Generation parameters (optimized for speed)
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=70,
        help="Maximum tokens to generate (default: 70, optimized for 9s limit)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (default: 0.3, lower for faster generation)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter (default: 0.9)"
    )
    parser.add_argument(
        "--time_limit",
        type=float,
        default=9.0,
        help="Time limit per answer in seconds (default: 9.0)"
    )
    
    # Input/Output
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to input questions JSON file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="answers.json",
        help="Path to output answers JSON file"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./model_cache",
        help="Directory to cache downloaded models (default: ./model_cache)"
    )
    
    args = parser.parse_args()
    
    # Load questions from input file
    if args.input_file is None:
        print("❌ Error: --input_file is required")
        print("Usage: python answer_model.py --input_file questions.json --output_file answers.json --load_in_8bit")
        sys.exit(1)
    
    print(f"📁 Loading questions from: {args.input_file}")
    questions = load_questions(args.input_file)
    print(f"✅ Loaded {len(questions)} questions")
    
    # Initialize model
    print("\n⚡ Initializing speed-optimized A-Agent...")
    answer_model = AAgent(
        model_name=args.model_name,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        cache_dir=args.cache_dir,
        time_limit_per_answer=args.time_limit
    )
    
    # Process questions
    print("\n🚀 Generating answers...")
    answers = answer_model.process_questions(
        questions=questions,
        output_file=args.output_file
    )
    
    print(f"\n✅ Completed! Generated {len(answers)} answers.")
    print(f"💾 Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()