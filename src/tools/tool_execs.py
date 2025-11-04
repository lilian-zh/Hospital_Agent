import os
import asyncio
import json
import time
from typing import Any, List, Optional
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from transformers import AutoModel
import subprocess
from pathlib import Path
from agents import RunContextWrapper, function_tool

from ..qdrant_collection import Qdrant_Collection

from .tool_handler import *
from .tool_request import *
from ..configs.agent_config import (
    QDRANT_ICD_PROCEDURE_COLLECTION_NAME,
    QDRANT_ICD_PROCEDURE_EMBEDDING_MODEL,
    EMBEDDING_MODEL_DIR,
    QDRANT_URL,
)



load_dotenv()


async def request_fetch_and_poll(
    patient_id: str,
    patient_hadm_id: str,
    handlers: List[ResourceHandler], 
    # return_raw: bool = False  # 👈 NEW FLAG
) -> Union[List[Any], str]:  # return raw results if flag is set
    logger.info(f"Requesting and polling results for patient {patient_id}")

    notes = []
    # raw_results = []

    for handler in handlers:
        try:
            result = await handler.fetch_result(patient_id, patient_hadm_id)

            result_resource = await handler.generate_result_resource(result, patient_id)

            if isinstance(result_resource, list):
                for res in result_resource:
                    if "note" in res and res["note"]:
                        notes.append(res["note"])
            else:
                if "note" in result_resource and result_resource["note"]:
                    notes.append(result_resource["note"])

        except Exception as e:
            logger.error(f"Error in handler execution: {e}")

    logger.info(f"Completed request_and_poll for patient {patient_id}")

    # if return_raw:
    #     return raw_results
    return notes



################################################################################

@function_tool
async def request_blood_test(
    wrapper: RunContextWrapper[PatientContext],
    lab_values: List[BloodValue], #[LabRequest],
    # **kwargs,
) -> str:
    """
    Requests specified blood tests for a patient and returns the results.

    Args:
        wrapper (RunContextWrapper[PatientContext]): Contains patient context and identifiers.
        lab_values (List[BloodValue]): Blood tests to request (e.g., "Hemoglobin", "Glucose").

    Returns:
        str: Combined results of requested blood tests.
    """

    # 🔧 fix the issue that input is a string
    if isinstance(lab_values, str):
        lab_values = json.loads(lab_values)

    logger.info("Requesting lab values ... 🩸🩸🩸")

    patient_data = wrapper.context.patient_data
    patient_hadm_id = wrapper.context.patient_hadm_id
    patient_id = wrapper.context.patient_id
    
    patient_data_table = patient_data.lab_events
    # print('patient_data_table:', patient_data_table)

    # Step 1: Create LabRequest instances
    lab_requests = [
        LabRequest(lab_value=lab_val)
        for lab_val in lab_values  # lab_val 是 str
    ]
    # lab_requests = [val.value for val in lab_values]
    # lab_requests = lab_values#.lab_values
    # print('lab_requests:', lab_requests)

    # Step 2: Create LabRequestHandler instances
    lab_handlers = [
        LabRequestHandler(
            request=lab_request,
            patient_data=patient_data_table,
        )
        for lab_request in lab_requests
    ]
    # print('lab_handlers:', lab_handlers)

    # Step 3: Call the asynchronous request_and_poll function
    notes = await request_fetch_and_poll(
        patient_id=patient_id,
        patient_hadm_id=patient_hadm_id,
        handlers=lab_handlers,
    )
    # print('notes_after:', notes)

    # Combine notes for return
    return "\n".join(notes) if isinstance(notes, list) else notes




################################################################################

@function_tool
async def request_urine_test(
    wrapper: RunContextWrapper[PatientContext],
    urine_values: List[UrineValue],
    # **kwargs,
) -> str:
    """
    Requests specified urine tests for a patient and returns the results.

    Args:
        wrapper (RunContextWrapper[PatientContext]): Contains patient context and identifiers.
        urine_values (List[UrineValue]): Urine tests to request (e.g., "Protein", "Glucose").

    Returns:
        str: Combined results of requested urine tests.

    Example:
        >>> request_urine_test(wrapper, ["Protein", "Glucose"])
        "Protein: Negative\\nGlucose: Trace"
    """

    if isinstance(urine_values, str):
        urine_values = json.loads(urine_values)

    logger.info("Requesting urine values ... 🫗💧🍻")

    patient_data = wrapper.context.patient_data
    patient_hadm_id = wrapper.context.patient_hadm_id
    patient_id = wrapper.context.patient_id

    patient_data_table = patient_data.lab_events

    urine_requests = [
        UrineRequest(urine_value=urine_val)
        for urine_val in urine_values
    ]
    # urine_requests = urine_values#.urine_values

    # 使用新的 handler(无 to_fhir)
    urine_handlers = [
        UrineRequestHandler(
            request=urine_request,
            patient_data=patient_data_table,
        )
        for urine_request in urine_requests
    ]

    # 发起请求，获取 notes(非 FHIR observation, 而是 string)
    notes = await request_fetch_and_poll(
        patient_id=patient_id,
        patient_hadm_id=patient_hadm_id,
        handlers=urine_handlers
    )

    return "\n".join(notes) if isinstance(notes, list) else notes


################################################################################

@function_tool
async def prescribe_medication(
    wrapper: RunContextWrapper[PatientContext],
    medications: List[MedicationRequest],
    # **kwargs,  # catch unused arguments
) -> str:
    """
    Prescribes specified medications for a patient and returns prescription details.

    Args:
        wrapper (RunContextWrapper[PatientContext]): Contains patient context and identifiers.
        medications (List[MedicationRequest]): Medications to prescribe with dosage information.

    Returns:
        str: Prescription confirmation and details.

    Example:
        >>> prescribe_medication(wrapper, [MedicationRequest(name="Amoxicillin", dosage="500mg", frequency="3 times daily")])
        "Prescription: Amoxicillin, 500mg, 3 times daily"
    """
    patient_data = wrapper.context.patient_data
    patient_hadm_id = wrapper.context.patient_hadm_id
    patient_id = wrapper.context.patient_id

    if isinstance(medications, str):
        try:
            medications = json.loads(medications)
            logger.warning("Medications parsed from string input.")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse medications string: {e}")
            return "Error: Invalid medications format. Please provide a valid list of medication dictionaries."

    patient_data_table = patient_data.medication  # Adjusted for medication data

    logger.info("Requesting medication values ... 💊💊💊")

    # Create MedicationRequest instances
    # medication_requests = [
    #     MedicationRequest(
    #         drug_name=params["drug_name"],
    #         dosage_text=params["dosage_text"],  # Ensure these fields are provided
    #         dosage_value=params["dosage_value"],
    #         dosage_unit=params["dosage_unit"],
    #         period=params["period"],
    #         period_unit=params["period_unit"],
    #         frequency=params["frequency"],
    #         route=params["route"],
    #         patient_id=patient_id,
    #     )
    #     for params in medications
    # ]
    medication_requests = medications
    # print('medication_requests:', medication_requests)

    # Create MedicationRequestHandler instances
    medication_handlers = [
        MedicationRequestHandler(
            request=medication_request,
            patient_data=patient_data_table,
        )
        for medication_request in medication_requests
    ]

    # Call the asynchronous request_fetch_and_poll function
    notes = await request_fetch_and_poll(
        patient_id=patient_id,
        patient_hadm_id=patient_hadm_id,
        handlers=medication_handlers,
    )

    return "\n".join(notes) if isinstance(notes, list) else notes


################################################################################

def connect_qdrant(embedding_client):
    """Connect to a Qdrant collection"""

    qdrant_client = QdrantClient(url=QDRANT_URL)
    collection = Qdrant_Collection(
        qdrant_client,
        embedding_client,
        QDRANT_ICD_PROCEDURE_COLLECTION_NAME,
        QDRANT_ICD_PROCEDURE_EMBEDDING_MODEL,
    )

    return qdrant_client, collection



@function_tool
async def search_procedure(
    wrapper: RunContextWrapper[PatientContext],
    procedure: str,
    # **kwargs,  # catch unused arguments
) -> str:
    """
    Searches for a specific medical procedure in the patient's records.

    Args:
        wrapper (RunContextWrapper[PatientContext]): Contains patient context and identifiers.
        procedure (str): Short description of the procedure to perform. This involves therapeutic procedures, like surgeries. For mostly diagnostic procedures like `ERCP` use the `request_radiology` tool.

    Returns:
        str: Search result or relevant procedure information.

    Example:
        >>> search_procedure(wrapper, "colonoscopy")
        "Procedure found: Colonoscopy"
    """

    logger.info("Processing procedure request ... 💉🏥🩹")
    logger.info("Loading Procedure Collections ...")

    patient_data = wrapper.context.patient_data
    patient_hadm_id = wrapper.context.patient_hadm_id
    patient_id = wrapper.context.patient_id

    embedding_client = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v3", cache_dir=EMBEDDING_MODEL_DIR, trust_remote_code=True,
    )
    try:
        qdrant_client, collection = connect_qdrant(embedding_client)
    except Exception as e:
        try:
            # Try to start Qdrant server
            subprocess.run(
                [
                    "docker",
                    "run",
                    "-p",
                    "6333:6333",
                    "-p",
                    "6334:6334",
                    "-v",
                    f"{os.getcwd()}/qdrant_storage:/qdrant/storage:z",
                    "qdrant/qdrant",
                ]
            )
            logger.info("Waiting for 10 seconds for Qdrant server to start ...")
            time.sleep(10)

            qdrant_client, collection = connect_qdrant(embedding_client)

        except Exception as e:
            logger.error(
                f"""Error loading procedure collection: {e}.
                Make sure to run
                    
                    `docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant`
                
                to start the Qdrant server on port 6333."""
            )
            raise e

    # Create ProcedureRequest instance
    procedure_request = ProcedureSearch(
        procedure=procedure,
    )
    # print(colored(f"Procedure Request: {procedure_request}", "yellow"))

    # Create ProcedureRequestHandler instance
    procedure_handler = ProcedureSearchRequestHandler(
        request=procedure_request,
        patient_data=patient_data,
        collection=collection,
    )
    # print(colored(f"Procedure Handler: {procedure_handler}", "yellow"))

    # Call the asynchronous request_fetch_and_poll function
    raw_results = await request_fetch_and_poll(
        patient_id=patient_id,
        patient_hadm_id=patient_hadm_id,
        handlers=[procedure_handler],
    )

    return "\n".join(raw_results) if isinstance(raw_results, list) else raw_results



@function_tool
async def request_procedure(
    wrapper: RunContextWrapper[PatientContext],
    procedure: str,
    # **kwargs,  # catch unused arguments
) -> str:
    """
    Requests a specific medical procedure for a patient and returns related notes.

    Args:
        wrapper (RunContextWrapper[PatientContext]): Contains patient context and identifiers.
        procedure (str): Exact name of the procedure to perform. Should be called after `search_procedure` tool with one of the options `option` where option is the exact name of the procedure. If the search did not return options you were looking for, try to search again, or skip. This involves therapeutic procedures, like surgeries. For mostly diagnostic procedures like `ERCP` use the `request_radiology` tool.

    Returns:
        str: Summary or notes related to the requested procedure.

    Example:
        >>> request_procedure(wrapper, "MRI")
        "MRI scheduled. Notes: Brain MRI with contrast."
    """

    logger.info("Processing procedure request ... 💉🏥🩹")
    logger.info("Loading Procedure Collections ...")

    patient_data = wrapper.context.patient_data
    patient_hadm_id = wrapper.context.patient_hadm_id
    patient_id = wrapper.context.patient_id

    # Create ProcedureRequest instance
    procedure_request = ProcedureRequest(
        procedure=procedure,
    )

    # Create ProcedureRequestHandler instance
    procedure_handler = ProcedureRequestHandler(
        request=procedure_request,
        patient_data=patient_data,
    )

    # Call the asynchronous request_fetch_and_poll function
    raw_results = await request_fetch_and_poll(
        patient_id=patient_id,
        patient_hadm_id=patient_hadm_id,
        handlers=[procedure_handler],
    )

    return "\n".join(raw_results) if isinstance(raw_results, list) else raw_results


################################################################################

@function_tool
async def request_physical_exam(
    wrapper: RunContextWrapper[PatientContext],
    # **kwargs,  # catch unused arguments
) -> str:
    """
    Requests a physical examination for a patient and returns the observations.

    Args:
        wrapper (RunContextWrapper[PatientContext]): Contains patient context and identifiers.

    Returns:
        str: Summary notes from the physical examination.

    Example:
        >>> request_physical_exam(wrapper)
        "Physical exam: BP 120/80, HR 72 bpm, no abnormalities detected."
    """

    logger.info("Requesting physical examination data ... 🩺🩺🩺")

    patient_data = wrapper.context.patient_data
    patient_hadm_id = wrapper.context.patient_hadm_id
    patient_id = wrapper.context.patient_id

    patient_data_table = patient_data.history_pe_admedication_diagnosis
    # print('patient_data_table:', patient_data_table)

    # Create PhysicalExamRequestFHIR instances
    pe_request = PhysicalExamRequest()
    # print('pe_request:', pe_request)

    # Create PhysicalExamRequestHandler instances
    pe_handler = PhysicalExamRequestHandler(
        request=pe_request,
        patient_data=patient_data_table,
    )
    # print('pe_handler:', pe_handler)
    # Call the asynchronous request_and_poll function
    notes = await request_fetch_and_poll(
        patient_id=patient_id,
        patient_hadm_id=patient_hadm_id,
        handlers=[pe_handler],
    )

    return "\n".join(notes) if isinstance(notes, list) else notes


################################################################################

@function_tool
async def request_microbiology(
    wrapper: RunContextWrapper[PatientContext],
    microbiology_tests: List[MicroBiologyValue],
    # **kwargs,  # catch unused arguments
) -> str:
    """
    Requests specified microbiology tests for a patient and returns the results.

    Args:
        wrapper (RunContextWrapper[PatientContext]): Contains patient context and identifiers.
        microbiology_tests (List[MicroBiologyValue]): Tests to request (e.g., "Blood culture", "Urine culture").

    Returns:
        str: Combined notes from microbiology test observations.

    Example:
        >>> request_microbiology(wrapper, ["Blood culture", "Sputum culture"])
        "Blood culture: No growth\\nSputum culture: Streptococcus pneumoniae detected"
    """

    # 🔧 修复输入为字符串的问题
    if isinstance(microbiology_tests, str):
        import json
        microbiology_tests = json.loads(microbiology_tests)

    logger.info("Requesting microbiology examination data ... 🦠🧫🧬")

    patient_data = wrapper.context.patient_data
    patient_hadm_id = wrapper.context.patient_hadm_id
    patient_id = wrapper.context.patient_id

    patient_data_table = (
        patient_data.microbiology
    ) 

    # Create MicrobiologyRequest instances
    microbiology_requests = [
        MicrobiologyRequest(microbiology_value=microbiology_test)
        for microbiology_test in microbiology_tests
    ]

    microbiology_handlers = [
        MicrobiologyRequestHandler(
            request=microbiology_request,
            patient_data=patient_data_table,
        )
        for microbiology_request in microbiology_requests
    ]

    # Call the asynchronous request_and_poll function
    notes = await request_fetch_and_poll(
        patient_id=patient_id,
        patient_hadm_id=patient_hadm_id,
        handlers=microbiology_handlers,
    )

    return "\n".join(notes) if isinstance(notes, list) else notes


################################################################################

@function_tool
async def request_radiology(
    wrapper: RunContextWrapper[PatientContext],
    modality: RadiologyModalityValue,
    region: RadiologyRegionValue,
    info: Optional[str] = None,
    # **kwargs,  # catch unused arguments
) -> str:
    """
    Requests a new radiology exam for a patient and returns the imaging report.

    Args:
        wrapper (RunContextWrapper[PatientContext]): Contains patient context and identifiers.
        modality (RadiologyModalityValue): Imaging modality to use (e.g., X-ray, CT, MRI).
        region (RadiologyRegionValue): Body region to image (e.g., Chest, Abdomen).
        info (Optional[str]): Additional clinical notes or questions for the exam.

    Returns:
        str: Summary from the radiology report.

    Example:
        >>> request_radiology(wrapper, modality="CT", region="Chest", info="Check for PE")
        "CT Chest: No pulmonary embolism detected."
    """

    logger.info("Requesting radiology examination data ... 🩻🩻🩻")

    patient_data = wrapper.context.patient_data
    patient_hadm_id = wrapper.context.patient_hadm_id
    patient_id = wrapper.context.patient_id

    patient_data_table = patient_data.radiology

    # Create RadiologyRequest instances
    radiology_request = RadiologyRequest(
        modality=modality,
        region=region,
        info=info or None,
        patient_id=patient_id,
    )

    # Create RadiologyRequestHandler instances
    radiology_handler = RadiologyRequestHandler(
        request=radiology_request,
        # patient_data=patient_data,
        patient_data=patient_data_table,
    )

    # Call the asynchronous request_and_poll function
    notes = await request_fetch_and_poll(
        patient_id=patient_id,
        patient_hadm_id=patient_hadm_id,
        handlers=[radiology_handler],
    )

    return "\n".join(notes) if isinstance(notes, list) else notes


################################################################################


@function_tool
def think(thought: str):
    """
    Used to reflect or plan before proceeding. This tool helps the model clarify its own reasoning before taking action.

    Args:
        thought (str): The assistant's internal reflection, reasoning, or plan before taking the next action.

    Returns:
        str: The thought.
    """
    return f"Internal reasoning: {thought}"



################################################################################

@function_tool
def talk(text: str):
    """
    Delivers a natural-language message to the patient. NEVER put the name, arguments, or results of a tool inside. NEVER simulate, paraphrase, or invent a tool call or its result.

    Args:
        text (str): Message content to speak to the patient. NEVER put a tool inside.

    Returns:
        str: The same message.

    Example:
        >>> talk("We will take your blood pressure now.")
        "We will take your blood pressure now."
    """
    return text



################################################################################

@function_tool
def finish(diagnosis: str) -> str:
    """
    Finalizes the patient case by recording the final diagnosis.

    Args:
        diagnosis (str): Final medical condition (e.g., "Acute Appendicitis"). Use concise terminology only.

    Returns:
        str: The confirmed diagnosis statement.
    """
    return diagnosis






