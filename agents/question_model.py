#!/usr/bin/python3

import time
import torch
import re
import random
import json
import argparse
import yaml
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.random.manual_seed(0)

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def sanitize_json_output(q: Any) -> Any:
    """Safely flattens JSON strings to prevent 'Invalid Control Character' errors."""
    if not isinstance(q, str):
        return q
    
    # Extract only the JSON dictionary part in case the model added extra text
    match = re.search(r'\{.*\}', q, re.DOTALL)
    if match:
        q = match.group(0)
        
    # Flatten ALL newlines, returns, and tabs into spaces so the JSON parser reads it as one single line
    q = q.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    return q


# ==========================================
# 1. QUESTION MODEL (QAgent)
# ==========================================

class QAgent(object):
    def __init__(self, **kwargs):
        ROOT_DIR = Path(__file__).parent.parent 
        model_path = ROOT_DIR / "hf_models" / "Qwen3-4B"
        
        if model_path.exists():
            model_name = str(model_path)
            print(f"✅ Loading Qwen model locally from: {model_name}")
        else:
            model_name = "Qwen/Qwen3-4B"
            print(f"⚠️ Local model not found. Attempting to load '{model_name}' from cache.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        
        print("⏳ Warming up GPU...")
        dummy_input = self.tokenizer("Hello", return_tensors="pt").to(self.model.device)
        self.model.generate(**dummy_input, max_new_tokens=2)
        print("✅ GPU Warm-up complete! Ready for fast generation.")

    def generate_response(
        self, message: str | List[str], system_prompt: Optional[str] = None, **kwargs
    ) -> str:
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        if isinstance(message, str):
            message = [message]
            
        all_messages = []
        for msg in message:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg},
            ]
            all_messages.append(messages)

        texts = []
        for messages in all_messages:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  
            )
            texts.append(text)

        model_inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        tgps_show_var = kwargs.get("tgps_show", False)
        
        if tgps_show_var:
            start_time = time.time()
            
        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", 400),
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": kwargs.get("do_sample", True),
        }
        
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = kwargs.get("temperature", 0.7)
            gen_kwargs["top_p"] = kwargs.get("top_p", 0.95)

        generated_ids = self.model.generate(
            **model_inputs,
            **gen_kwargs
        )
        
        if tgps_show_var:
            generation_time = time.time() - start_time

        batch_outs = []
        if tgps_show_var:
            token_len = 0
            
        for i, (input_ids, generated_sequence) in enumerate(
            zip(model_inputs.input_ids, generated_ids)
        ):
            output_ids = generated_sequence[len(input_ids) :].tolist()

            if tgps_show_var:
                token_len += len(output_ids)

            content = self.tokenizer.decode(
                output_ids, skip_special_tokens=True
            ).strip("\n")
            
            content = re.sub(r'<think>[\s\S]*?(?:</think>|$)', '', content).strip()
            
            batch_outs.append(content)
            
        if tgps_show_var:
            return (
                batch_outs[0] if len(batch_outs) == 1 else batch_outs,
                token_len,
                generation_time,
            )
        return batch_outs[0] if len(batch_outs) == 1 else batch_outs, None, None


# ==========================================
# 2. QUESTIONING AGENT WORKFLOW
# ==========================================

class QuestioningAgent(object):
    r"""Agent responsible for generating questions"""

    def __init__(self, **kwargs):
        self.agent = QAgent(**kwargs)

    def build_inc_samples(self, inc_samples: List[Dict[str, str]], topic: str) -> str:
        if not inc_samples:
            return ""
        fmt = (
            "EXAMPLE: {}\n"
            "{{\n"
            '  "topic": "{}",\n'
            '  "question": "{}",\n'
            '  "choices": ["A) {}", "B) {}", "C) {}", "D) {}"],\n'
            '  "answer": "{}",\n'
            '  "explanation": "{}"\n'
            "}}"
        )

        sample_str = ""
        for sample in inc_samples:
            question = sample.get("question", "")
            choices = sample.get("choices", [""] * 4)
            answer = sample.get("answer", "")
            explanation = sample.get("explanation", "")
            sample_str += (
                fmt.format(
                    topic, topic.split("/")[-1], question, *choices, answer, explanation
                )
                + "\n\n"
            )
        return sample_str.strip()

    def build_prompt(
        self,
        topic: str,
        wadvsys: bool = True,
        wicl: bool = True,
        inc_samples: List[Dict[str, str]] | None = None,
    ) -> Tuple[str, str]:
        if wadvsys:
            # RESTORED AND ENHANCED: Aggressive, high-difficulty system prompt with minimalist tactics
            sys_prompt = """
            You are a master of abstract logic and an elite examiner for the world's most difficult analytical assessments. 
            You design problems that are conceptually punishing, utilizing the absolute least amount of viable information. 
            In absolutely no way should you include more information than strictly necessary. 
            Your questions must require extensive thinking power, be linked by very thin threads, and often force the solver into trial-and-error reasoning where the solution is only visible to those with extreme lateral thinking skills.
            """
        else:
            sys_prompt = "You are an examiner tasked with creating extremely difficult multiple-choice questions."
            
        tmpl = (
            "Generate an EXTREMELY DIFFICULT, riddle-like MCQ on the topic: {0}.\n\n"
            "**CRITICAL REQUIREMENTS:**\n"
            '1.  **Topic Alignment**: The "question" must be relevant to {1}, but hidden behind dense logic.\n'
            "2.  **Minimalist Difficulty**: The problem must be nearly unsolvable. Use the bare minimum viable information and in absolutely no way include more data than necessary.\n"
            "3.  **Extensive Thinking**: The logic must be linked by very thin threads. Include trial-and-error dynamics where the solver must test the options to find the truth.\n"
            '4.  **Choices (4 total)**: Generate exactly FOUR options, labeled "A)", "B)", "C)", and "D)".\n'
            "5.  **Single Correct Answer**: Ensure that option {2} is the only logically sound answer despite the cryptic nature of the problem.\n"
            "6.  **Diabolical Distractors**: Options {3} must be extremely plausible traps that mislead even high-level thinkers.\n"
            '7.  **Answer Key**: The "answer" field in the JSON should be ONLY the letter {4}.\n'
            '8.  **Explanation**: Provide a clear but concise logical proof for the correct option.\n\n'
            "{5}"
            "RESPONSE FORMAT: Generate a valid JSON object as shown below.\n\n"
            "EXAMPLE: {6}\n"
            "{{\n"
            '  "topic": "{7}",\n'
            '  "question": "...",\n'
            '  "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],\n'
            '  "answer": "{8}",\n'
            '  "explanation": "..."\n'
            "}}"
        )
        correct_option = random.choice(["A", "B", "C", "D"])
        distractors = ", ".join(
            [opt for opt in ["A", "B", "C", "D"] if opt != correct_option]
        )

        if wicl:
            inc_samples_ex = self.build_inc_samples(inc_samples, topic)
        else:
            inc_samples_ex = ""
        prompt = tmpl.format(
            topic,
            topic,
            correct_option,
            distractors,
            correct_option,
            inc_samples_ex,
            topic,
            topic.split("/")[-1],
            correct_option,
            correct_option,
        )

        return prompt, sys_prompt

    def generate_question(
        self,
        topic: Tuple[str, str] | List[Tuple[str, str]],
        wadvsys: bool,
        wicl: bool,
        inc_samples: Dict[str, List[Dict[str, str]]] | None,
        **gen_kwargs,
    ) -> Tuple[List[str], int | None, float | None]:
        if isinstance(topic, list):
            prompt = []
            for t in topic:
                p, sp = self.build_prompt(
                    f"{t[0]}/{t[1]}", wadvsys, wicl, inc_samples[t[1]]
                )
                prompt.append(p)
        else:
            prompt, sp = self.build_prompt(
                f"{topic[0]}/{topic[1]}", wadvsys, wicl, inc_samples[topic[1]]
            )

        resp, tl, gt = self.agent.generate_response(prompt, sp, **gen_kwargs)

        if (
            isinstance(resp, list) and all(isinstance(r, str) for r in resp)
        ) or isinstance(resp, str):
            return resp, tl, gt
        else:
            return (
                "",
                tl,
                gt if not isinstance(resp, list) else [""] * len(resp),
                tl,
                gt,
            )

    def generate_batches(
        self,
        num_questions: int,
        topics: Dict[str, List[str]],
        batch_size: int = 5,
        wadvsys: bool = True,
        wicl: bool = True,
        inc_samples: Dict[str, List[Dict[str, str]]] | None = None,
        **kwargs,
    ) -> Tuple[List[str], List[int | None], List[float | None]]:
        extended_topics = self.populate_topics(topics, num_questions)
        questions = []
        tls, gts = [], []
        total_batches = (len(extended_topics) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="STEPS: ")

        for i in range(0, len(extended_topics), batch_size):
            batch_topics = extended_topics[i : i + batch_size]
            batch_questions = self.generate_question(
                batch_topics, wadvsys, wicl, inc_samples, **kwargs
            )
            questions.extend(batch_questions[0]), tls.append(
                batch_questions[1]
            ), gts.append(batch_questions[2])
            pbar.update(1)
            
        if len(extended_topics) % batch_size != 0:
            batch_topics = extended_topics[-(len(extended_topics) % batch_size) :]
            batch_questions = self.generate_question(
                batch_topics, wadvsys, wicl, inc_samples, **kwargs
            )
            questions.extend(batch_questions[0]), tls.append(
                batch_questions[1]
            ), gts.append(batch_questions[2])
            pbar.update(1)
        pbar.close()
        return questions, tls, gts

    def count_tokens_q(self, text: str) -> int:
        if not hasattr(self.agent, "tokenizer"):
            raise AttributeError("The agent does not have a tokenizer attribute.")
        return len(self.agent.tokenizer.encode(text, add_special_tokens=False))

    def filter_questions(
        self, questions: List[Union[str, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        def basic_checks(q2: Dict[str, Any]) -> bool:
            required_keys = ["topic", "question", "choices", "answer"]
            if all((key in q2) for key in required_keys):
                checks = all(
                    isinstance(choice, str)
                    and len(choice) > 2
                    and choice[0].upper() in "ABCD"
                    for choice in q2["choices"]
                )
                if (
                    isinstance(q2["choices"], list)
                    and len(q2["choices"]) == 4
                    and checks
                ):
                    # STRICT TOKEN FILTERING MATCHING JUDGING CRITERIA
                    check_len = sum(self.count_tokens_q(str(q2[k])) for k in ['question', 'answer'])
                    check_len += sum(self.count_tokens_q(str(choice)) for choice in q2['choices']) - 15
                    
                    if check_len < 130:
                        if check_len + self.count_tokens_q(str(q2.get('explanation', 'None'))) <= 1024:
                            if isinstance(q2['answer'], str) and len(q2['answer']) == 1 and q2['answer'].upper() in "ABCD":
                                return True
            return False

        correct_format_question = []
        for i, q in enumerate(questions):
            if isinstance(q, dict):
                if basic_checks(q):
                    correct_format_question.append(q)
            elif isinstance(q, str):
                clean_q = sanitize_json_output(q)
                try:
                    q1 = json.loads(clean_q, strict=False)
                    if basic_checks(q1):
                        correct_format_question.append(q1)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON at index {i}: {q}")
                    continue
            else:
                continue
                
        # Judging criteria requires a 50% pass rate
        if len(correct_format_question) >= 0.5 * len(questions):
            return correct_format_question
        return list()

    def save_questions(self, questions: Any, file_path: str | Path) -> None:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(questions, f, indent=4)

    def populate_topics(
        self, topics: Dict[str, List[str]], num_questions: int
    ) -> List[str]:
        if not isinstance(topics, dict):
            raise ValueError(
                "Topics must be a dictionary with topic names as keys and lists of subtopics as values."
            )
        all_subtopics = [(t, st) for t, sublist in topics.items() for st in sublist]
        if not all_subtopics:
            raise ValueError("No subtopics found in the provided topics dictionary.")
        selected_topics = random.choices(all_subtopics, k=num_questions)
        return selected_topics

    @staticmethod
    def load_icl_samples(file_path: str | Path) -> Dict[str, List[Dict[str, str]]]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        with open(file_path, "r") as f:
            samples = json.load(f)
        if not isinstance(samples, dict):
            raise ValueError("Samples must be inside dictionary.")
        return samples


# ==========================================
# 3. MAIN EXECUTION SCRIPT
# ==========================================

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Generate questions using the QuestioningAgent."
    )
    argparser.add_argument(
        "--num_questions",
        type=int,
        default=10,
        help="Total number of questions to generate.",
    )
    argparser.add_argument(
        "--output_file",
        type=str,
        default="outputs/questions.json",
        help="Output file name to save the generated questions.",
    )
    argparser.add_argument(
        "--batch_size", type=int, default=5, help="Batch size for generating questions."
    )
    argparser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output for debugging."
    )
    args = argparser.parse_args()

    ROOT_DIR = Path(__file__).parent.parent
    
    inc_samples_path = ROOT_DIR / "assets" / "topics_example.json"
    topics_path = ROOT_DIR / "assets" / "topics.json"
    qgen_path = ROOT_DIR / "qgen.yaml"

    inc_samples = QuestioningAgent.load_icl_samples(inc_samples_path)

    with open(topics_path) as f:
        topics = json.load(f)

    agent = QuestioningAgent()
    gen_kwargs = {"tgps_show": True}
    
    if qgen_path.exists():
        with open(qgen_path, "r") as f:
            gen_kwargs.update(yaml.safe_load(f))

    # RESTORED: controlled sampling for unique, creative problems
    gen_kwargs["do_sample"] = True
    gen_kwargs["temperature"] = 0.7  
    gen_kwargs["top_p"] = 0.95
    
    if "repetition_penalty" in gen_kwargs:
        del gen_kwargs["repetition_penalty"]

    question, tls, gts = agent.generate_batches(
        num_questions=args.num_questions,
        topics=topics,
        batch_size=args.batch_size,
        wadvsys=True,
        wicl=True,
        inc_samples=inc_samples,
        **gen_kwargs,
    )
    print(f"\nGenerated {len(question)} questions!")
    
    if args.verbose:
        for q in question:
            print(q, flush=True)
        print("\n" + "=" * 50 + "\n")
        if gen_kwargs.get("tgps_show", False):
            print("Time taken per batch generation:", gts)
            print("Tokens generated per batch:", tls)
            print(
                f"Total Time Taken: {sum(gts):.3f} seconds; Total Tokens: {sum(tls)}; TGPS: {sum(tls)/sum(gts):.3f} seconds\n"
            )
        print("+" * 50 + "\n")

    ques = []
    for q in question:
        try:
            clean_q = sanitize_json_output(q)
            q_dict = json.loads(clean_q, strict=False)
            q = q_dict  
        except json.JSONDecodeError as e:
            prompt = (
                "Extract **ONLY** the topic, question, choices, answer, and explanation while discarding the rest.\n"
                "Also please remove JSON code block text with backticks** like **```json** and **```**.\n\n"
                "String:\n"
                "{}\n\n"
                "Given Format:\n"
                "{{\n"
                '  "topic": "...",\n'
                '  "question": "...",\n'
                '  "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],\n'
                '  "answer": "Only the option letter (A, B, C, or D)",\n'
                '  "explanation": "..."\n'
                "}}"
            )
            fallback_q = agent.agent.generate_response(
                prompt.format(q),
                "You are an expert JSON extractor.",
                max_new_tokens=400,
                do_sample=False,
            )
            clean_fallback = sanitize_json_output(fallback_q)
            try:
                q = json.loads(clean_fallback, strict=False)
            except Exception as e_fallback:
                print(f"Fallback extraction failed. Skipping invalid JSON.\nOriginal Error: {e}")
                q = clean_fallback 
        ques.append(q)
        
    final_output_path = ROOT_DIR / args.output_file
    filtered_output_path = ROOT_DIR / args.output_file.replace(
        "questions.json", "filtered_questions.json"
    )
    
    agent.save_questions(ques, final_output_path)
    agent.save_questions(agent.filter_questions(ques), filtered_output_path)
    
    print(f"✅ Unfiltered questions saved safely to: {final_output_path}")
    print(f"✅ Filtered questions saved safely to: {filtered_output_path}")