from agents import Agent, Runner
import math
import json
from typing import Any
from pydantic import Field
import re

from .tools.tool_request import PatientContext
from .mes_collect import EvaluationOutputCollector as OutputCollector
    

async def run_simulation(
    doctor: Agent,
    patient: Agent,
    primary_symptom: str,
    context: PatientContext, 
    completion_prompt: str,
    output_collector: Any = Field(default_factory=OutputCollector),
    max_steps: int = 10,
):
    """
    Simulate conversation between doctor and patient using OpenAI Agents SDK
    start from the patient, then the doctor, alternatively, until the doctor call finish or reaches max_steps
    """

    runner = Runner()

    starter = (
        f"My {primary_symptom}" if primary_symptom.strip()
        else "Hi, I'm not feeling well but not sure how to describe it. Could you help me figure out what's wrong?"
    )
    speaker = doctor
    listener = patient
    doctor_turns = 0
    doctor_mes = [{"role": "user", "content": starter}]
    patient_mes = [{"role": "assistant", "content": starter}]

    finish_called = False
    force_wrapup = False

    output_collector.display_message(name=patient.name, message=starter)
    while not finish_called:
        if speaker.name.lower().startswith("doctor"):
            # wrap-up or not?
            if doctor_turns >= max_steps and not force_wrapup:
                doctor_mes.append({"role": "system", "content": completion_prompt})
                force_wrapup = True
            elif force_wrapup and not finish_called:
                doctor.tools = [t for t in doctor.tools if t.name.lower() == "finish"]
                doctor.model_settings.tool_choice="required"

            result = await runner.run(input=doctor_mes, starting_agent=speaker, context=context, max_turns=20)
        else:
            result = await runner.run(input=patient_mes, starting_agent=speaker, context=context, max_turns=20)

        new_input_items = [item.to_input_item() for item in result.new_items]
        new_message = re.sub(r"<think>.*?</think>", "", result.final_output, flags=re.DOTALL).strip()

        tool_calls_pending_output = {} 

        for item in new_input_items:
            item_type = item.get("type")
            if item_type in ["function_call", "function_call_output"]:
                item_for_history = item
                if item_type == "function_call":
                    item_for_history = item.copy()

                    if 'logprobs' in item_for_history:
                        del item_for_history['logprobs']

                    call_id = item.get("call_id")
                    function_name = item.get("name")
                    
                    if function_name == "finish":
                        finish_called = True
                    
                    if call_id:
                        tool_calls_pending_output[call_id] = item
                # 3. 将最终确定的 item_for_history 添加到 doctor_mes
                doctor_mes.append(item_for_history)

        # 第二次遍历：合并工具调用和输出，并进行最终记录
        for item in new_input_items:
            item_type = item.get("type")

            if item_type == "function_call_output":
                call_id = item.get("call_id")
                if call_id in tool_calls_pending_output:
                    call_item = tool_calls_pending_output[call_id]
                    
                    function_name = call_item.get("name")
                    arguments = call_item.get("arguments")
                    output = item.get("output")

                    output_collector.display_action(
                        message=output,
                        function_name=function_name,
                        arguments=arguments
                    )
                    
                    del tool_calls_pending_output[call_id]

        for call_id, call_item in tool_calls_pending_output.items():
            if not finish_called:
                 output_collector.display_action(
                    message="Tool call occurred, but no output was received in this turn.",
                    function_name=call_item.get("name"),
                    arguments=call_item.get("arguments")
                )

        if finish_called:
            break

        if new_message:
            output_collector.display_message(name=speaker.name, message=new_message)
            if speaker.name.lower().startswith("doctor"):
                doctor_mes.append({"role": "assistant", "content": new_message})
                patient_mes.append({"role": "user", "content": new_message})
            else:
                doctor_mes.append({"role": "user", "content": new_message})
                patient_mes.append({"role": "assistant", "content": new_message})

        if speaker.name.lower().startswith("doctor"):
            doctor_turns += 1

        # force doctor to summarize under wrap-up
        if force_wrapup:
            speaker = doctor
            listener = patient
        else:
            speaker, listener = listener, speaker
