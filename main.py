from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_classic.output_parsers import (
    StructuredOutputParser,
    ResponseSchema
)


# ==================================================
# 1. Load Model
# ==================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


# ==================================================
# 2. Generate Text
# ==================================================

def generate_text(prompt, max_new_tokens=250):

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return response.strip()


# ==================================================
# 3. Get Prompt From User
# ==================================================

print("\n======================================")
print("       AI PROMPT OPTIMIZER")
print("======================================")

user_prompt = input("\nEnter your prompt: ").strip()


# ==================================================
# 4. CHAIN 1 - Analyze Prompt
# ==================================================

analysis_prompt = PromptTemplate(
    template="""
You are an AI Prompt Analyzer.

Analyze the user's prompt.

Do NOT answer the user's request.

Identify:

Role:
Task:
Context:
Constraints:
Output Format:

IMPORTANT:
- Keep the original topic exactly the same.
- Do not narrow the topic.
- Do not invent a different task.
- Keep the analysis short.

User Prompt:
{user_prompt}
""",
    input_variables=["user_prompt"]
)


def run_analysis(data):

    prompt = analysis_prompt.format(
        user_prompt=data["user_prompt"]
    )

    analysis = generate_text(
        prompt,
        max_new_tokens=150
    )

    return {
        "user_prompt": data["user_prompt"],
        "analysis": analysis
    }


analysis_chain = RunnableLambda(run_analysis)


# ==================================================
# 5. CHAIN 2 - Optimize Prompt
# ==================================================

optimization_prompt = PromptTemplate(
    template="""
You are a professional AI Prompt Optimizer.

Your ONLY job is to rewrite the original prompt into
a clearer and more useful PROMPT.

Original Prompt:
{user_prompt}

Analysis:
{analysis}

IMPORTANT RULES:

1. Keep the EXACT same topic.
2. Keep the EXACT same main task.
3. Do NOT answer the original prompt.
4. Do NOT change the topic.
5. Do NOT narrow the topic to one part unless the original prompt does so.
6. Do NOT invent unrelated information.
7. Do NOT add unnecessary features.
8. Make the prompt clearer and more specific.
9. Keep the same target audience.
10. The result must be a PROMPT that another AI can follow.
11. The result must NOT be the answer.
12. Return ONLY the optimized prompt.
13. Do NOT use JSON.
14. Do NOT use quotation marks around the whole prompt.
15. Do NOT use triple quotes.

Example:

Original:
Explain cloud computing to a beginner

Good:
Explain cloud computing to a complete beginner.
Start with a simple definition, use an everyday analogy,
give 2-3 real-world examples, avoid technical jargon,
and organize the explanation with clear headings and bullet points.

Bad:
Cloud computing is a technology that allows users to
store and access data over the internet.

Now optimize the original prompt.
""",
    input_variables=[
        "user_prompt",
        "analysis"
    ]
)


def run_optimization(data):

    prompt = optimization_prompt.format(
        user_prompt=data["user_prompt"],
        analysis=data["analysis"]
    )

    optimized_prompt = generate_text(
        prompt,
        max_new_tokens=180
    )

    return {
        "user_prompt": data["user_prompt"],
        "analysis": data["analysis"],
        "optimized_prompt": optimized_prompt
    }


optimization_chain = RunnableLambda(run_optimization)


# ==================================================
# 6. Output Parser Schemas
# ==================================================

role_schema = ResponseSchema(
    name="role",
    description="The role that the AI should take."
)

task_schema = ResponseSchema(
    name="task",
    description="The main task that the AI should perform."
)

context_schema = ResponseSchema(
    name="context",
    description="Useful context about the request."
)

constraints_schema = ResponseSchema(
    name="constraints",
    description="Important requirements or limitations."
)

output_format_schema = ResponseSchema(
    name="output_format",
    description="The desired format of the final answer."
)

optimized_prompt_schema = ResponseSchema(
    name="optimized_prompt",
    description="The improved prompt. It must be a prompt, not an answer."
)


response_schemas = [
    role_schema,
    task_schema,
    context_schema,
    constraints_schema,
    output_format_schema,
    optimized_prompt_schema
]


output_parser = StructuredOutputParser.from_response_schemas(
    response_schemas
)

format_instructions = output_parser.get_format_instructions()


# ==================================================
# 7. Final Structured Output Chain
# ==================================================

final_prompt = PromptTemplate(
    template="""
Create a JSON object using the information below.

Original Prompt:
{user_prompt}

Analysis:
{analysis}

Optimized Prompt:
{optimized_prompt}

IMPORTANT:

- Return ONLY valid JSON.
- Use double quotes for JSON strings.
- Do NOT use triple quotes.
- Do NOT use markdown code fences.
- Do NOT put line breaks inside JSON string values.
- Keep optimized_prompt as a PROMPT, not an answer.
- Do not change the original topic.

Required JSON format:

{format_instructions}
""",
    input_variables=[
        "user_prompt",
        "analysis",
        "optimized_prompt",
        "format_instructions"
    ]
)


def create_final_result(data):

    prompt = final_prompt.format(
        user_prompt=data["user_prompt"],
        analysis=data["analysis"],
        optimized_prompt=data["optimized_prompt"],
        format_instructions=format_instructions
    )

    return generate_text(
        prompt,
        max_new_tokens=300
    )


final_chain = RunnableLambda(create_final_result)


# ==================================================
# 8. Connect Chains
# ==================================================

full_chain = (
    analysis_chain
    | optimization_chain
    | final_chain
)


# ==================================================
# 9. Run Full Chain
# ==================================================

raw_response = full_chain.invoke({
    "user_prompt": user_prompt
})


# ==================================================
# 10. Extract JSON
# ==================================================

start = raw_response.find("{")
end = raw_response.rfind("}")


if start != -1 and end != -1:

    json_text = raw_response[start:end + 1]

    try:

        result = output_parser.parse(json_text)

        print("\n======================================")
        print("          ANALYSIS RESULT")
        print("======================================")

        print("\nOriginal Prompt:")
        print(user_prompt)

        print("\nRole:")
        print(result["role"])

        print("\nTask:")
        print(result["task"])

        print("\nContext:")
        print(result["context"])

        print("\nConstraints:")
        print(result["constraints"])

        print("\nOutput Format:")
        print(result["output_format"])

        print("\nOptimized Prompt:")
        print(result["optimized_prompt"])

        print("\n======================================")
        print("       CHAINING SUCCESS")
        print("======================================")

        print("\n       OUTPUT PARSER SUCCESS")
        print("======================================")

    except Exception as e:

        print("\n❌ Output Parser Error:")
        print(e)

        print("\nRaw JSON:")
        print(json_text)

else:

    print("\n❌ No JSON object found.")

    print("\nRaw AI Response:")
    print(raw_response)