````python
import os
import re
import subprocess
import tempfile
from typing import TypedDict

from fastapi import FastAPI
from pydantic import BaseModel, Field

from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langserve import add_routes


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=API_KEY,
    temperature=0.2,
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Verilog Generator",
    version="1.0.0",
    description="Generate, test and simulate Verilog HDL using Gemini."
)


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class VerilogRequest(BaseModel):
    task: str = Field(
        ...,
        description="Description of the digital circuit to design."
    )


class VerilogResponse(BaseModel):
    task: str
    verilog_code: str
    testbench: str
    simulation_result: str


# ============================================================
# LANGGRAPH STATE
# ============================================================

class VerilogState(TypedDict, total=False):
    task: str
    verilog_code: str
    testbench: str
    simulation_result: str


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_code(text: str) -> str:
    """Remove Markdown code fences from Gemini output."""

    text = text.strip()

    text = re.sub(
        r"```(?:verilog|systemverilog|sv)?",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = text.replace("```", "")

    return text.strip()


def response_to_text(response) -> str:
    """Extract text from a LangChain Gemini response."""

    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])

            else:
                parts.append(str(item))

        return "\n".join(parts)

    return str(content)


# ============================================================
# LANGGRAPH NODE 1
# GENERATE VERILOG
# ============================================================

def developer_node(state: VerilogState) -> VerilogState:

    task = state["task"]

    prompt = f"""
You are an expert digital electronics and Verilog HDL engineer.

Design the following digital circuit:

{task}

Requirements:

1. Generate synthesizable Verilog HDL.
2. Use standard Verilog syntax.
3. Include a clear module declaration.
4. Use sensible input and output names.
5. Do not include explanations.
6. Do not use Markdown code fences.
7. Return ONLY the Verilog source code.
"""

    response = llm.invoke(prompt)

    verilog_code = clean_code(
        response_to_text(response)
    )

    return {
        "verilog_code": verilog_code
    }


# ============================================================
# LANGGRAPH NODE 2
# GENERATE TESTBENCH
# ============================================================

def tester_node(state: VerilogState) -> VerilogState:

    task = state["task"]
    verilog_code = state["verilog_code"]

    prompt = f"""
You are an expert Verilog verification engineer.

Circuit requirement:

{task}

Verilog design:

{verilog_code}

Create a complete Verilog testbench.

Requirements:

1. Instantiate the design correctly.
2. Test important input combinations.
3. Include reset behavior if applicable.
4. Include clock generation if required.
5. Use $display or $monitor to show results.
6. Use $finish to end the simulation.
7. The testbench must compile with Icarus Verilog.
8. Do not include Markdown code fences.
9. Return ONLY the testbench code.
"""

    response = llm.invoke(prompt)

    testbench = clean_code(
        response_to_text(response)
    )

    return {
        "testbench": testbench
    }


# ============================================================
# LANGGRAPH NODE 3
# COMPILE AND SIMULATE VERILOG
# ============================================================

def simulator_node(state: VerilogState) -> VerilogState:

    verilog_code = state["verilog_code"]
    testbench = state["testbench"]

    try:

        with tempfile.TemporaryDirectory() as temp_dir:

            design_file = os.path.join(
                temp_dir,
                "design.v"
            )

            testbench_file = os.path.join(
                temp_dir,
                "testbench.v"
            )

            output_file = os.path.join(
                temp_dir,
                "simulation.out"
            )

            # Write Verilog design
            with open(
                design_file,
                "w",
                encoding="utf-8"
            ) as file:

                file.write(verilog_code)

            # Write testbench
            with open(
                testbench_file,
                "w",
                encoding="utf-8"
            ) as file:

                file.write(testbench)

            # Compile
            compile_process = subprocess.run(
                [
                    "iverilog",
                    "-g2012",
                    "-o",
                    output_file,
                    design_file,
                    testbench_file
                ],
                capture_output=True,
                text=True,
                timeout=30
            )

            if compile_process.returncode != 0:

                return {
                    "simulation_result":
                    "COMPILATION FAILED\n\n"
                    + compile_process.stderr
                }

            # Simulate
            simulation_process = subprocess.run(
                ["vvp", output_file],
                capture_output=True,
                text=True,
                timeout=30
            )

            if simulation_process.returncode != 0:

                return {
                    "simulation_result":
                    "SIMULATION FAILED\n\n"
                    + simulation_process.stderr
                }

            output = simulation_process.stdout.strip()

            if not output:

                output = (
                    "Simulation completed successfully "
                    "with no console output."
                )

            return {
                "simulation_result":
                "SIMULATION SUCCESSFUL\n\n"
                + output
            }

    except FileNotFoundError:

        return {
            "simulation_result":
            "Icarus Verilog is not installed on the server."
        }

    except subprocess.TimeoutExpired:

        return {
            "simulation_result":
            "Simulation timed out after 30 seconds."
        }

    except Exception as error:

        return {
            "simulation_result":
            f"Simulation error: {error}"
        }


# ============================================================
# LANGGRAPH WORKFLOW
# ============================================================

workflow = StateGraph(VerilogState)

workflow.add_node(
    "developer",
    developer_node
)

workflow.add_node(
    "tester",
    tester_node
)

workflow.add_node(
    "simulator",
    simulator_node
)

workflow.add_edge(
    START,
    "developer"
)

workflow.add_edge(
    "developer",
    "tester"
)

workflow.add_edge(
    "tester",
    "simulator"
)

workflow.add_edge(
    "simulator",
    END
)

verilog_graph = workflow.compile()


# ============================================================
# LANGSERVE ADAPTER
# ============================================================

def run_verilog_pipeline(request):

    if isinstance(request, VerilogRequest):

        task = request.task

    elif isinstance(request, dict):

        task = request.get("task", "")

    else:

        task = str(request)

    task = task.strip()

    if not task:

        raise ValueError(
            "Task description cannot be empty."
        )

    result = verilog_graph.invoke(
        {
            "task": task
        }
    )

    return VerilogResponse(

        task=task,

        verilog_code=result.get(
            "verilog_code",
            ""
        ),

        testbench=result.get(
            "testbench",
            ""
        ),

        simulation_result=result.get(
            "simulation_result",
            ""
        )
    )


# ============================================================
# LANGSERVE ROUTABLE CHAIN
# ============================================================

verilog_chain = RunnableLambda(
    run_verilog_pipeline
).with_types(
    input_type=VerilogRequest,
    output_type=VerilogResponse
)


# ============================================================
# LANGSERVE ROUTES
# ============================================================

add_routes(
    app,
    verilog_chain,
    path="/verilog"
)


# ============================================================
# FASTAPI ROUTES
# ============================================================

@app.get("/")
def root():

    return {
        "service": "AI Verilog Generator",
        "status": "running",
        "langserve": "/verilog",
        "invoke": "/verilog/invoke",
        "docs": "/docs"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "gemini_configured": bool(API_KEY)
    }


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000"))
    )
````
