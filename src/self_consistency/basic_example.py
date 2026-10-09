from pathlib import Path

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Solve step by step: John has 5 apples, gives away 2, and buys twice as many apples as he has left. How many apples does he have in total?",
    "Solve for x: 3x + 5 = 20.",
    "Solve step by step: A notebook costs $4. Dante buys 3 notebooks and pays with a $20 bill. How much change should he receive?",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)


def main():
    # Create an LLM.
    llm = LLM(
        model="Qwen/Qwen2.5-Math-1.5B-Instruct",
        gpu_memory_utilization=0.75,
        max_model_len=1024,
        max_num_seqs=1,
        max_num_batched_tokens=1024,
        enforce_eager=True,
    )
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    saved_outputs = []
    markdown_outputs = ["# Generated Outputs"]
    for index, output in enumerate(outputs, start=1):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)
        saved_outputs.append(f"Prompt: {prompt}\nOutput:\n{generated_text}\n" + "-" * 60)
        # Guardar los prompts y outputs en formato Markdown
        markdown_prompt = prompt.replace("$", r"\$")
        markdown_outputs.append(
            f"## Question {index}\n\n{markdown_prompt}\n\n"
            f"### Answer\n\n{generated_text.strip()}"
        )

    output_path = Path(__file__).with_name("basic_example_output.txt")
    output_path.write_text("\n\n".join(saved_outputs) + "\n", encoding="utf-8")
    print(f"Outputs saved to: {output_path}")
    markdown_path = output_path.with_suffix(".md")
    markdown_path.write_text("\n\n".join(markdown_outputs) + "\n", encoding="utf-8")
    print(f"Markdown saved to: {markdown_path}")


if __name__ == "__main__":
    main()
