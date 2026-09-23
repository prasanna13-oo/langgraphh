import sys
import traceback
import subprocess
import tempfile
import os

from typing import TypedDict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

import google.generativeai as genai
from google.colab import userdata


# ==========================================
# 1. LLM INITIALIZATION
# ==========================================

try:
    api_key = userdata.get("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    print("API Key configured successfully.")
except userdata.SecretNotFoundError:
    print("Error: GEMINI_API_KEY not found in Colab secrets")
    api_key = None

if not api_key:
    raise ValueError("GEMINI_API_KEY is required.")

llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key
)
llm = llm_flash


# ==========================================
# 2. STATE DEFINITION
# ==========================================

class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    testbench: Optional[str]
    report: Optional[str]


# ==========================================
# 3. VERILOG TOOLS
# ==========================================

@tool
def run_verilog_code(code: str, testbench: str = "") -> str:
    """Compile and simulate Verilog using Icarus Verilog."""

    clean_code = (
        str(code)
        .replace("```verilog", "")
        .replace("```v", "")
        .replace("```", "")
        .strip()
    )

    clean_testbench = (
        str(testbench)
        .replace("```verilog", "")
        .replace("```v", "")
        .replace("```", "")
        .strip()
    )

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            verilog_file = os.path.join(temp_dir, "design.v")
            testbench_file = os.path.join(temp_dir, "testbench.v")
            output_file = os.path.join(temp_dir, "simulation")

            with open(verilog_file, "w") as f:
                f.write(clean_code)

            if clean_testbench:
                with open(testbench_file, "w") as f:
                    f.write(clean_testbench)

                compile_result = subprocess.run(
                    [
                        "iverilog",
                        "-o",
                        output_file,
                        verilog_file,
                        testbench_file
                    ],
                    capture_output=True,
                    text=True
                )
            else:
                compile_result = subprocess.run(
                    [
                        "iverilog",
                        "-o",
                        output_file,
                        verilog_file
                    ],
                    capture_output=True,
                    text=True
                )

            if compile_result.returncode != 0:
                return (
                    "VERILOG COMPILATION ERROR:\n"
                    + compile_result.stderr
                )

            if not clean_testbench:
                return "Verilog compilation successful."

            simulation_result = subprocess.run(
                ["vvp", output_file],
                capture_output=True,
                text=True
            )

            if simulation_result.returncode != 0:
                return (
                    "VERILOG SIMULATION ERROR:\n"
                    + simulation_result.stderr
                )

            output = simulation_result.stdout.strip()

            if output:
                return (
                    "VERILOG SIMULATION SUCCESSFUL:\n\n"
                    + output
                )

            return (
                "Verilog simulation completed successfully "
                "with no terminal output."
            )

    except FileNotFoundError:
        return (
            "Icarus Verilog was not found. "
            "Install it using: apt-get install iverilog"
        )
    except Exception:
        return "Execution Error:\n" + traceback.format_exc()


@tool
def generate_verilog_testbench(task_description: str) -> str:
    """Generate a Verilog testbench for the requested digital design."""

    prompt = f"""
You are a Senior Digital Design Verification Engineer.

The user wants to implement the following digital logic design:

{task_description}

Generate a complete Verilog testbench.

Requirements:
1. Use Verilog HDL.
2. Instantiate the expected DUT.
3. Test normal cases.
4. Test edge cases where applicable.
5. Apply appropriate input combinations.
6. Use $display statements to show results.
7. Use $monitor if useful.
8. Include $finish.
9. Make it compatible with Icarus Verilog.
10. Return ONLY the Verilog testbench code.
11. Do NOT use markdown code fences.
"""

    response = llm.invoke(prompt)
    content = response.content

    if isinstance(content, list):
        return (
            content[0].get("text", "")
            if isinstance(content[0], dict)
            else str(content[0])
        )

    return str(content)


# ==========================================
# 4. GRAPH NODES
# ==========================================

def task_input_node(state: CrewState):
    print("\n" + "=" * 50)
    print("--- NEW VERILOG TASK INITIALIZATION ---")

    user_task = input(
        "Enter the digital design task "
        "(or type 'exit' to quit): "
    ).strip()

    if user_task.lower() == "exit":
        return {"next_step": "exit"}

    return {
        "messages": [HumanMessage(content=user_task)],
        "next_step": "developer"
    }


def real_time_developer(state: CrewState):
    print("\n[Developer] Generating Verilog HDL using Gemini...")

    task = state["messages"][-1].content

    dev_prompt = f"""
You are an expert Digital Design Engineer.

Convert the following requirement into a synthesizable Verilog HDL design:

TASK:
{task}

Requirements:
1. Use Verilog HDL.
2. Create a clear top-level module.
3. Define appropriate input and output ports.
4. Use synthesizable Verilog.
5. Use always blocks or assign statements where appropriate.
6. Do not use Python.
7. Do not use a testbench.
8. Return ONLY the Verilog code.
9. Do NOT use markdown code fences.
10. Make the design compatible with Icarus Verilog.
"""

    response = llm_flash.invoke(dev_prompt)
    content = response.content

    if isinstance(content, list):
        code_str = (
            content[0].get("text", "")
            if isinstance(content[0], dict)
            else str(content[0])
        )
    else:
        code_str = str(content)

    code_str = (
        code_str
        .replace("```verilog", "")
        .replace("```v", "")
        .replace("```", "")
        .strip()
    )

    print("\n--- GENERATED VERILOG ---\n")
    print(code_str)

    return {"code": code_str}


def real_time_tester(state: CrewState):
    print("\n[Tester] Generating testbench and simulating Verilog...")

    task = state["messages"][-1].content

    testbench = generate_verilog_testbench.invoke(task)

    testbench = (
        testbench
        .replace("```verilog", "")
        .replace("```v", "")
        .replace("```", "")
        .strip()
    )

    print("\n--- GENERATED TESTBENCH ---\n")
    print(testbench)

    execution_result = run_verilog_code.invoke(
        {
            "code": state["code"],
            "testbench": testbench
        }
    )

    report = (
        "### GENERATED VERILOG:\n"
        + state["code"]
        + "\n\n### TESTBENCH:\n"
        + testbench
        + "\n\n### SIMULATION RESULT:\n"
        + execution_result
    )

    return {
        "testbench": testbench,
        "report": report
    }


def manager_decision_node(state: CrewState):
    print("\n" + "=" * 50)
    print("--- MANAGER DASHBOARD : VERILOG TEST REPORT ---")
    print(state.get("report", "No report available."))
    print("=" * 50)

    user_input = input(
        "\nCommand (store / another): "
    ).lower().strip()

    if user_input == "store":
        return {"next_step": "archiver"}

    return {"next_step": "task_input"}


def archiver_node(state: CrewState):
    print(
        "\n[Archiver] Verilog design stored successfully. "
        "Closing workflow."
    )
    return {"next_step": "exit"}


# ==========================================
# 5. GRAPH CONSTRUCTION
# ==========================================

rt_workflow = StateGraph(CrewState)

rt_workflow.add_node("task_input", task_input_node)
rt_workflow.add_node("developer", real_time_developer)
rt_workflow.add_node("tester", real_time_tester)
rt_workflow.add_node("manager_decision", manager_decision_node)
rt_workflow.add_node("archiver", archiver_node)

rt_workflow.add_edge(START, "task_input")


def route_from_input(state):
    if state.get("next_step") == "exit":
        return END
    return "developer"


rt_workflow.add_conditional_edges(
    "task_input",
    route_from_input
)

rt_workflow.add_edge("developer", "tester")
rt_workflow.add_edge("tester", "manager_decision")


def route_from_decision(state):
    if state.get("next_step") == "archiver":
        return "archiver"
    return "task_input"


rt_workflow.add_conditional_edges(
    "manager_decision",
    route_from_decision
)

rt_workflow.add_edge("archiver", END)

rt_app = rt_workflow.compile()

print(
    "Interactive Verilog generation pipeline "
    "compiled and ready for live execution."
)


# ==========================================
# 6. EXECUTION
# ==========================================

if __name__ == "__main__":
    try:
        rt_app.invoke(
            {"messages": []},
            config={"recursion_limit": 50}
        )

    except KeyboardInterrupt:
        print("\nStopped by user.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
