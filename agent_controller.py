import argparse
import copy
import itertools
import re
import time
from datetime import datetime
from pathlib import Path
from config import LocalModelConfig
from prompts import ROLE_PROMPTS, CONVERSATION_PROMPT, FINAL_SUMMARY_PROMPT
from model_utils import (
    flush_cuda_cache,
    load_model_and_tokenizer,
    generate_text,
    generate_text_batch,
    generate_text_stream,
    HiddenReasoningFilter,
    strip_hidden_reasoning,
)


class Agent:
    def __init__(self, name, system_prompt, specialty=""):
        self.name = name
        self.system_prompt = system_prompt
        self.specialty = specialty
        self.history = []

    def build_messages(self, question, discussion):
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": CONVERSATION_PROMPT.format(
                question=question,
                discussion=discussion or "None yet.",
            )},
        ]

    def answer(self, question, discussion, model, tokenizer, config, seed=None, hidden_callback=None):
        prompt = tokenizer.apply_chat_template(
            self.build_messages(question, discussion),
            tokenize=False,
            add_generation_prompt=True,
        )
        generated = generate_text(model, tokenizer, prompt, config, seed=seed)
        visible, hidden = strip_hidden_reasoning(generated)
        if hidden_callback and hidden:
            hidden_callback(hidden)
        self.history.append(visible)
        return visible

    def answer_stream(self, question, discussion, model, tokenizer, config, generation_kwargs=None, hidden_callback=None, seed=None):
        prompt = tokenizer.apply_chat_template(
            self.build_messages(question, discussion),
            tokenize=False,
            add_generation_prompt=True,
        )
        streamer, _ = generate_text_stream(
            model,
            tokenizer,
            prompt,
            config,
            generation_kwargs=generation_kwargs,
            seed=seed,
        )
        filterer = HiddenReasoningFilter()
        generated = ""
        for chunk in streamer:
            visible_chunks = filterer.feed(chunk)
            for visible_chunk in visible_chunks:
                generated += visible_chunk
                yield visible_chunk
        for visible_chunk in filterer.flush():
            generated += visible_chunk
            yield visible_chunk

        hidden = filterer.get_hidden()
        if hidden_callback and hidden:
            hidden_callback(hidden)
        self.history.append(generated)
        return generated

        generated = ""
        for chunk in streamer:
            generated += chunk
            yield chunk
        self.history.append(generated)
        return generated


def normalize_answer(text):
    return " ".join(text.lower().strip().split())


def consensus_reached(responses):
    normalized = [normalize_answer(r) for r in responses]
    if len(set(normalized)) == 1:
        return True

    token_sets = [set(re.findall(r"\w+", text)) for text in normalized]
    if not all(token_sets):
        return False

    common_tokens = set.intersection(*token_sets)
    smallest_set = min(len(tokens) for tokens in token_sets)
    if smallest_set == 0:
        return False

    overlap = len(common_tokens) / smallest_set
    return overlap >= 0.75


class DebateOrchestrator:
    def __init__(self, config):
        self.config = config
        self.model, self.tokenizer = load_model_and_tokenizer(config)
        print(f"Model loaded, tokenizer name: {self.tokenizer.__class__.__name__}")
        self.agents = [Agent(role["name"], role["system"], role.get("specialty", "")) for role in ROLE_PROMPTS]

    def run_debate(self, question, seed=None, skip_summary=False):
        print(f"Starting debate for question: {question}")
        discussion = ""
        round_replies = []
        for round_index in range(self.config.debate_rounds):
            round_replies = []
            print(f"\n--- Beginning round {round_index + 1} (batching {len(self.agents)} agents) ---")
            prompts = [
                self.tokenizer.apply_chat_template(
                    agent.build_messages(question, discussion),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for agent in self.agents
            ]
            responses = generate_text_batch(self.model, self.tokenizer, prompts, self.config, seed=seed)
            filtered_responses = []
            for response in responses:
                visible, _ = strip_hidden_reasoning(response)
                filtered_responses.append(visible)
            for agent, response in zip(self.agents, filtered_responses):
                agent.history.append(response)
                round_replies.append(f"{agent.name} ({agent.specialty}): {response}")
            discussion = "\n".join(round_replies)
            # Truncate discussion to prevent the prompt growing beyond the context window.
            max_chars = getattr(self.config, "max_discussion_chars", 4000)
            if len(discussion) > max_chars:
                discussion = discussion[-max_chars:]
            print(f"\n=== Round {round_index + 1} replies ===")
            for reply in round_replies:
                print(reply)
            if consensus_reached([reply.split(":", 1)[1].strip() for reply in round_replies]):
                print("\nConsensus reached after round", round_index + 1)
                break

        if skip_summary:
            print("Skipping final summary generation for benchmark mode.")
            return None, round_replies

        final_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": FINAL_SUMMARY_PROMPT.format(replies="\n\n".join(round_replies))}],
            tokenize=False,
            add_generation_prompt=True,
        )
        final_answer_raw = generate_text(
            self.model,
            self.tokenizer,
            final_prompt,
            self.config,
            generation_kwargs={
                "max_new_tokens": self.config.summary_max_new_tokens + self.config.summary_extra_tokens if self.config.soft_token_limit else self.config.summary_max_new_tokens,
                "temperature": self.config.summary_temperature,
                "top_p": getattr(self.config, "top_p", 0.95),
                "top_k": getattr(self.config, "top_k", 50),
                "num_beams": self.config.summary_beams,
                "do_sample": getattr(self.config, "summary_do_sample", False),
                "early_stopping": True,
                "no_repeat_ngram_size": getattr(self.config, "no_repeat_ngram_size", 2),
            },
            seed=seed,
        )
        final_answer, _ = strip_hidden_reasoning(final_answer_raw)
        return final_answer, round_replies

    def run_debate_stream(self, question, ui_callback, seed=None):
        ui_callback("system", f"> Question received: {question}\n\n")
        discussion = ""
        round_replies = []
        for round_index in range(self.config.debate_rounds):
            round_replies = []
            ui_callback("system", f"\n========================================\n")
            ui_callback("system", f"Beginning round {round_index + 1}\n")
            ui_callback("system", f"========================================\n\n")

            for agent in self.agents:
                ui_callback("agent_name", f"{agent.name} ")
                ui_callback("agent_specialty", f"({agent.specialty}):")
                ui_callback("system", "\n")
                full_reply = ""
                generation_kwargs = {"max_new_tokens": self.config.agent_max_new_tokens + self.config.agent_extra_tokens} if self.config.soft_token_limit else {"max_new_tokens": self.config.agent_max_new_tokens}
                for chunk in agent.answer_stream(
                    question,
                    discussion,
                    self.model,
                    self.tokenizer,
                    self.config,
                    generation_kwargs=generation_kwargs,
                    hidden_callback=lambda hidden: ui_callback("hidden_reasoning", hidden),
                    seed=seed,
                ):
                    ui_callback("agent_text", chunk)
                    full_reply += chunk
                ui_callback("system", "\n\n")
                round_replies.append(f"{agent.name} ({agent.specialty}): {full_reply}")

            discussion = "\n".join(round_replies)
            max_chars = getattr(self.config, "max_discussion_chars", 4000)
            if len(discussion) > max_chars:
                discussion = discussion[-max_chars:]

            if consensus_reached([reply.split(":", 1)[1].strip() for reply in round_replies]):
                ui_callback("system", f"\nAgreement reached after round {round_index + 1}.\n")
                break

        ui_callback("system", "\n========================================\n")
        ui_callback("system", "Preparing final summary:\n")
        ui_callback("system", "========================================\n\n")
        final_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": FINAL_SUMMARY_PROMPT.format(replies="\n\n".join(round_replies))}],
            tokenize=False,
            add_generation_prompt=True,
        )
        filterer = HiddenReasoningFilter()
        streamer, _ = generate_text_stream(
            self.model,
            self.tokenizer,
            final_prompt,
            self.config,
            generation_kwargs={
                "max_new_tokens": self.config.summary_max_new_tokens + self.config.summary_extra_tokens if self.config.soft_token_limit else self.config.summary_max_new_tokens,
                "temperature": getattr(self.config, "summary_temperature", 0.3),
                "top_p": getattr(self.config, "top_p", 0.95),
                "num_beams": 1,
                "do_sample": getattr(self.config, "summary_do_sample", False),
            },
            seed=seed,
        )

        ui_callback("summary_title", "SUMMARY:\n")
        for chunk in streamer:
            for visible_chunk in filterer.feed(chunk):
                ui_callback("summary_text", visible_chunk)
        for visible_chunk in filterer.flush():
            ui_callback("summary_text", visible_chunk)
        hidden = filterer.get_hidden()
        if hidden:
            ui_callback("hidden_reasoning", hidden)

        ui_callback("system", "\n\nDebate complete.\n")
        ui_callback("done", None)

    def run_debate_batch(self, question, ui_callback, seed=None):
        ui_callback("system", f"> Question received: {question}\n\n")
        discussion = ""
        round_replies = []
        for round_index in range(self.config.debate_rounds):
            round_replies = []
            ui_callback("system", f"\n========================================\n")
            ui_callback("system", f"Beginning round {round_index + 1}\n")
            ui_callback("system", f"========================================\n\n")

            prompts = [
                self.tokenizer.apply_chat_template(
                    agent.build_messages(question, discussion),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for agent in self.agents
            ]
            responses = generate_text_batch(self.model, self.tokenizer, prompts, self.config, seed=seed)
            for agent, response in zip(self.agents, responses):
                visible, hidden = strip_hidden_reasoning(response)
                if hidden:
                    ui_callback("hidden_reasoning", hidden)
                ui_callback("agent_name", f"{agent.name} ")
                ui_callback("agent_specialty", f"({agent.specialty}):")
                ui_callback("agent_text", visible + "\n\n")
                round_replies.append(f"{agent.name} ({agent.specialty}): {visible}")

            discussion = "\n".join(round_replies)
            max_chars = getattr(self.config, "max_discussion_chars", 4000)
            if len(discussion) > max_chars:
                discussion = discussion[-max_chars:]

            if consensus_reached([reply.split(":", 1)[1].strip() for reply in round_replies]):
                ui_callback("system", f"\nAgreement reached after round {round_index + 1}.\n")
                break

        ui_callback("system", "\n========================================\n")
        ui_callback("system", "Preparing final summary:\n")
        ui_callback("system", "========================================\n\n")
        final_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": FINAL_SUMMARY_PROMPT.format(replies="\n\n".join(round_replies))}],
            tokenize=False,
            add_generation_prompt=True,
        )
        final_answer_raw = generate_text(
            self.model,
            self.tokenizer,
            final_prompt,
            self.config,
            generation_kwargs={
                "max_new_tokens": self.config.summary_max_new_tokens + self.config.summary_extra_tokens if self.config.soft_token_limit else self.config.summary_max_new_tokens,
                "temperature": getattr(self.config, "summary_temperature", 0.3),
                "top_p": getattr(self.config, "top_p", 0.95),
                "num_beams": 1,
                "do_sample": getattr(self.config, "summary_do_sample", False),
            },
            seed=seed,
        )
        final_answer, final_hidden = strip_hidden_reasoning(final_answer_raw)
        if final_hidden:
            ui_callback("hidden_reasoning", final_hidden)
        ui_callback("summary_title", "SUMMARY:\n")
        ui_callback("summary_text", final_answer)
        ui_callback("system", "\n\nDebate complete.\n")
        ui_callback("done", None)


def _format_flag_description(flags):
    parts = []
    if flags.get("use_torch_compile"):
        parts.append("torch.compile")
    if flags.get("use_bfloat16"):
        parts.append("bfloat16")
    attn_impl = flags.get("attn_implementation")
    if attn_impl is not None and attn_impl != "sdpa":
        parts.append(f"attn={attn_impl or 'default'}")
    if not parts:
        return "baseline"
    return "+".join(parts)


def _build_benchmark_cases():
    flag_names = [
        "use_torch_compile",
        "use_bfloat16",
    ]

    cases = [
        {
            "name": "baseline",
            "flags": {**{name: False for name in flag_names}, "attn_implementation": "sdpa"},
        }
    ]

    for name in flag_names:
        cases.append(
            {
                "name": name,
                "flags": {**{flag: False for flag in flag_names}, name: True, "attn_implementation": "sdpa"},
            }
        )

    cases.append(
        {
            "name": "all_flags",
            "flags": {**{name: True for name in flag_names}, "attn_implementation": "sdpa"},
        }
    )

    return cases


def _write_benchmark_markdown(path, question, seed, results):
    header = [
        "# Benchmark results",
        "",
        f"- question: `{question}`",
        f"- seed: {seed}",
        f"- generated: {datetime.utcnow().isoformat()} UTC",
        "",
        "| case | flags | load time (s) | run time (s) | total time (s) | status | note |",
        "|---|---|---|---|---|---|---|",
    ]
    rows = []
    for result in results:
        rows.append(
            "| {} | {} | {:.4f} | {:.4f} | {:.4f} | {} | {} |".format(
                result["name"],
                result["flags_desc"],
                result.get("load_time", 0.0),
                result.get("run_time", 0.0),
                result.get("total_time", 0.0),
                "ok" if result.get("success", False) else "failed",
                result.get("note", ""),
            )
        )
    Path(path).write_text("\n".join(header + rows), encoding="utf-8")


def run_benchmark(question, base_config, output_path, seed):
    cases = _build_benchmark_cases()
    results = []
    for case in cases:
        config = copy.deepcopy(base_config)
        for key, value in case["flags"].items():
            setattr(config, key, value)

        case_name = case["name"]
        flags_desc = _format_flag_description(case["flags"])
        config.debate_rounds = 1
        print(f"\nRunning benchmark case: {case_name} ({flags_desc})")
        start_load = time.perf_counter()
        orchestrator = None
        try:
            orchestrator = DebateOrchestrator(config)
            load_time = time.perf_counter() - start_load
            start_run = time.perf_counter()
            orchestrator.run_debate(question, seed=seed, skip_summary=True)
            run_time = time.perf_counter() - start_run
            total_time = load_time + run_time
            results.append(
                {
                    "name": case_name,
                    "flags_desc": flags_desc,
                    "load_time": load_time,
                    "run_time": run_time,
                    "total_time": total_time,
                    "success": True,
                    "note": "",
                }
            )
        except Exception as exc:
            note = str(exc).replace("|", "\\|")
            results.append(
                {
                    "name": case_name,
                    "flags_desc": flags_desc,
                    "load_time": 0.0,
                    "run_time": 0.0,
                    "total_time": 0.0,
                    "success": False,
                    "note": note,
                }
            )
            print(f"Benchmark case failed: {note}")
        finally:
            if orchestrator is not None:
                orchestrator.model = None
                orchestrator.tokenizer = None
                del orchestrator
            flush_cuda_cache()

    _write_benchmark_markdown(output_path, question, seed, results)


def main():
    parser = argparse.ArgumentParser(description="Run a local socialist multi-agent debate.")
    parser.add_argument("question", type=str, help="The question to ask the local socialist agents.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional local path or Hugging Face model ID to load as the base model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional override for max new tokens per response.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Optional override for debate rounds.",
    )
    parser.add_argument(
        "--use-torch-compile",
        action="store_true",
        help="Wrap model with torch.compile() for JIT optimization. Adds startup overhead; best for repeated use.",
    )
    parser.add_argument(
        "--use-bfloat16",
        action="store_true",
        help="Use bfloat16 compute dtype if the GPU supports it. Falls back to float16 otherwise.",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        help="Attention backend to use. Options: 'sdpa' (default), 'eager', or '' for model default.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark mode and write timing results to a markdown file.",
    )
    parser.add_argument(
        "--benchmark-output",
        type=str,
        default="benchmark_results.md",
        help="Path to the markdown file for benchmark results.",
    )
    parser.add_argument(
        "--benchmark-seed",
        type=int,
        default=42,
        help="Random seed used for benchmark generation consistency.",
    )
    args = parser.parse_args()

    config = LocalModelConfig()
    if args.model:
        config.base_model = args.model
    if args.max_tokens is not None:
        config.max_new_tokens = args.max_tokens
    if args.rounds is not None:
        config.debate_rounds = args.rounds
    if args.use_torch_compile:
        config.use_torch_compile = True
    if args.use_bfloat16:
        config.compute_dtype_bfloat16 = True
    if args.attn_implementation is not None:
        config.attn_implementation = args.attn_implementation

    if args.benchmark:
        benchmark_results = run_benchmark(args.question, config, args.benchmark_output, args.benchmark_seed)
        print(f"Benchmark complete. Results written to {args.benchmark_output}")
        return

    orchestrator = DebateOrchestrator(config)
    final_answer, replies = orchestrator.run_debate(args.question)

    print("\n=== Final agreed answer ===")
    print(final_answer)


if __name__ == "__main__":
    main()
