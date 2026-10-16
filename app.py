from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
import json
import re

app = Flask(__name__)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading AI model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto"
)

print("AI model loaded successfully!")


# =========================================================
# AI TEXT GENERATION
# =========================================================

def generate_text(prompt, max_new_tokens=120):

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    if text.startswith(prompt):
        text = text[len(prompt):]

    return text.strip()


# =========================================================
# OUTPUT PARSER
# =========================================================

response_schemas = [

    ResponseSchema(
        name="role",
        description="The role of the AI."
    ),

    ResponseSchema(
        name="task",
        description="The main task."
    ),

    ResponseSchema(
        name="context",
        description="Relevant context."
    ),

    ResponseSchema(
        name="constraints",
        description="Important constraints."
    ),

    ResponseSchema(
        name="output_format",
        description="Expected output format."
    ),

    ResponseSchema(
        name="optimized_prompt",
        description="The final improved prompt."
    )
]

output_parser = StructuredOutputParser.from_response_schemas(
    response_schemas
)


# =========================================================
# CHAIN 1 — PROMPT ANALYSIS
# =========================================================

analysis_prompt = PromptTemplate.from_template(
"""
You are a prompt analyzer.

Analyze the following user prompt:

{prompt}

DO NOT answer it.

Return ONLY JSON.

Use exactly:

{{
  "role": "User",
  "task": "...",
  "context": "...",
  "constraints": "...",
  "output_format": "..."
}}

Rules:

- Keep the same topic.
- Task describes what the user wants.
- Context contains useful information from the prompt.
- Constraints contains limitations mentioned by the user.
- If no constraints exist, write "Not available".
- If no output format is mentioned, write "Not available".
- Do not answer the prompt.
"""
)


def run_analysis(data):

    prompt = analysis_prompt.format(
        prompt=data["prompt"]
    )

    raw = generate_text(
        prompt,
        max_new_tokens=150
    )

    start = raw.find("{")
    end = raw.rfind("}")

    if start != -1 and end != -1:

        try:

            analysis = json.loads(
                raw[start:end + 1]
            )

        except Exception:

            analysis = {
                "role": "User",
                "task": data["prompt"],
                "context": "Not available",
                "constraints": "Not available",
                "output_format": "Not available"
            }

    else:

        analysis = {
            "role": "User",
            "task": data["prompt"],
            "context": "Not available",
            "constraints": "Not available",
            "output_format": "Not available"
        }

    return {
        "original_prompt": data["prompt"],
        "analysis": analysis
    }


analysis_chain = RunnableLambda(
    run_analysis
)


# =========================================================
# CHAIN 2 — IMPROVEMENT INSTRUCTIONS
# =========================================================

improvement_prompt = PromptTemplate.from_template(
"""
You are a Prompt Improvement Assistant.

Your ONLY job is to suggest short instructions
that can improve the user's prompt.

DO NOT answer the user's prompt.

Original prompt:
{original_prompt}

Task:
{task}

Return ONLY 2 or 3 short improvement instructions.

IMPORTANT:

- Do not write the answer.
- Do not write an email.
- Do not write code.
- Do not explain the topic.
- Do not add unrelated requirements.
- Each instruction must be one short sentence.
- Do not use headings.
- Do not use Markdown.
- Do not number the instructions.

For an email-writing prompt, useful instructions include:

Use a professional and polite tone.
Clearly state the request.
Keep the email concise.
Include the relevant reason or context.

For programming prompts, useful instructions include:

Make the code beginner-friendly.
Include a short explanation.
Include a simple example input and output.

For explanation prompts, useful instructions include:

Start with a simple definition.
Use an everyday analogy.
Give real-world examples.
Avoid unnecessary jargon.

For study-plan prompts, useful instructions include:

Break the plan into clear daily goals.
Make the schedule realistic for a beginner.
Include practice activities.
Include a simple progress check.

For comparison prompts, useful instructions include:

Compare the main features of both options.
Highlight key similarities and differences.
Present the comparison in a clear structure.
Include the main advantages and disadvantages of each option.
"""
)


def run_improvements(data):

    analysis = data["analysis"]

    prompt = improvement_prompt.format(
        original_prompt=data["original_prompt"],
        task=analysis.get(
            "task",
            data["original_prompt"]
        )
    )

    improvements = generate_text(
        prompt,
        max_new_tokens=80
    )

    return {
        "original_prompt": data["original_prompt"],
        "analysis": analysis,
        "improvements": improvements
    }


improvement_chain = RunnableLambda(
    run_improvements
)


# =========================================================
# CLEAN IMPROVEMENTS
# =========================================================

def clean_improvements(text):

    text = text.replace("```", "")

    lines = text.splitlines()

    cleaned = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        line = re.sub(
            r"^\d+[\.\)\-]\s*",
            "",
            line
        )

        line = re.sub(
            r"^[-*]\s*",
            "",
            line
        )

        lower = line.lower()

        if (
            lower.startswith("cloud computing is")
            or lower.startswith("the answer is")
            or lower.startswith("cloud computing allows")
            or lower.startswith("the average is")
            or lower.startswith("the code")
            or lower.startswith("dear ")
            or lower.startswith("subject:")
            or lower.startswith("to:")
        ):
            continue

        if len(line.split()) > 18:
            continue

        cleaned.append(line)

    return cleaned[:3]


# =========================================================
# BUILD OPTIMIZED PROMPT
# =========================================================

def build_optimized_prompt(data):

    original_prompt = data["original_prompt"]

    analysis = data["analysis"]

    improvements = clean_improvements(
        data["improvements"]
    )

    task = analysis.get(
        "task",
        original_prompt
    )

    context = analysis.get(
        "context",
        "Not available"
    )

    output_format = analysis.get(
        "output_format",
        "Not available"
    )

    if (
        not output_format
        or output_format.strip().lower() == "not available"
    ):
        output_format = "markdown"

    lower_prompt = original_prompt.lower()


    # =====================================================
    # EXPLANATION PROMPT
    # =====================================================

    if (
        lower_prompt.startswith("explain ")
        and "beginner" in lower_prompt
    ):

        optimized_prompt = (

            f"{original_prompt.rstrip('.')}."
            f" Start with a simple definition."
            f" Use an everyday analogy."
            f" Provide 2-3 real-world examples."
            f" Avoid unnecessary technical jargon."
            f" Present the explanation in clear {output_format}."
        )

        return optimized_prompt


    # =====================================================
    # PROGRAMMING PROMPT
    # =====================================================

    programming_action_words = [
        "write code",
        "write a program",
        "create code",
        "create a program",
        "build a program",
        "develop a program",
        "code for",
        "implement",
        "debug",
        "fix this code",
        "modify this code",
        "python code",
        "javascript code",
        "java code",
        "c++ code",
        "c# code"
    ]

    is_programming = any(
        phrase in lower_prompt
        for phrase in programming_action_words
    )


    if is_programming:

        optimized_prompt = (

            f"{original_prompt.rstrip('.')}."
            f" Write clear and beginner-friendly code."
            f" Include a short explanation of how the code works."
            f" Provide a simple example input and output."
            f" Present the response in clear {output_format}."
        )

        return optimized_prompt


    # =====================================================
    # EMAIL PROMPT
    # =====================================================

    email_keywords = [
        "email",
        "e-mail",
        "mail"
    ]

    is_email = any(
        word in lower_prompt
        for word in email_keywords
    )


    if is_email:

        optimized_prompt = (

            f"{original_prompt.rstrip('.')}."
            f" Use a professional and polite tone."
            f" Clearly state the request."
            f" Briefly include the relevant context."
            f" Keep the email concise."
            f" Present the response in clear {output_format}."
        )

        return optimized_prompt


    # =====================================================
    # STUDY PLAN PROMPT
    # =====================================================

    study_plan_keywords = [
        "study plan",
        "learning plan",
        "study schedule",
        "learning schedule",
        "30 day plan",
        "30-day plan",
        "30 day study plan",
        "30-day study plan"
    ]

    is_study_plan = any(
        phrase in lower_prompt
        for phrase in study_plan_keywords
    )


    if is_study_plan:

        optimized_prompt = (

            f"{original_prompt.rstrip('.')}."
            f" Break the plan into clear daily goals."
            f" Make the schedule realistic for a beginner."
            f" Include practice activities."
            f" Include a simple progress check."
            f" Present the response in clear {output_format}."
        )

        return optimized_prompt


    # =====================================================
    # COMPARISON PROMPT
    # =====================================================

    comparison_keywords = [
        "compare",
        "comparison",
        "compare between",
        "difference between",
        "differences between",
        "similarities between",
        "pros and cons"
    ]

    is_comparison = any(
        phrase in lower_prompt
        for phrase in comparison_keywords
    )


    if is_comparison:

        optimized_prompt = (

            f"{original_prompt.rstrip('.')}."
            f" Compare the main features of both options."
            f" Highlight key similarities and differences."
            f" Include the main advantages and disadvantages of each option."
            f" Present the comparison in a clear structure."
            f" Present the response in clear {output_format}."
        )

        return optimized_prompt


    # =====================================================
    # GENERAL PROMPT
    # =====================================================

    optimized_prompt = task

    if (
        context
        and context.strip().lower() != "not available"
    ):

        optimized_prompt += (
            f" Consider the following context: {context}."
        )

    if improvements:

        optimized_prompt += " " + " ".join(

            improvement.rstrip(".") + "."

            for improvement in improvements
        )

    optimized_prompt += (

        f" Present the response in {output_format}."
    )

    return optimized_prompt


# =========================================================
# FINAL RESULT
# =========================================================

def build_result(data):

    analysis = data["analysis"]

    optimized_prompt = build_optimized_prompt(
        data
    )

    output_format = analysis.get(
        "output_format",
        "Not available"
    )

    if (
        not output_format
        or output_format.strip().lower() == "not available"
    ):

        output_format = "markdown"


    result = {

        "role": analysis.get(
            "role",
            "User"
        ),

        "task": analysis.get(
            "task",
            data["original_prompt"]
        ),

        "context": analysis.get(
            "context",
            "Not available"
        ),

        "constraints": analysis.get(
            "constraints",
            "Not available"
        ),

        "output_format": output_format,

        "optimized_prompt": optimized_prompt

    }

    return result


# =========================================================
# FULL CHAIN
# =========================================================

full_chain = (

    analysis_chain
    | improvement_chain
    | RunnableLambda(build_result)

)


# =========================================================
# FLASK ROUTES
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route(
    "/optimize",
    methods=["POST"]
)
def optimize():

    try:

        data = request.get_json()

        user_prompt = data.get(
            "prompt",
            ""
        ).strip()

        if not user_prompt:

            return jsonify({
                "error": "Please enter a prompt."
            }), 400

        result = full_chain.invoke({
            "prompt": user_prompt
        })

        return jsonify(result)

    except Exception as e:

        print(
            "ERROR:",
            e
        )

        return jsonify({
            "error": str(e)
        }), 500


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    print(
        "Starting AI Prompt Optimizer..."
    )

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )