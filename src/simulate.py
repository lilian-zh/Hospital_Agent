import os
import re
import glob
import logging
import asyncio
from typing import List, Optional, Tuple
from dataclasses import dataclass
from agents import Agent, ModelSettings, set_tracing_disabled
from agents.agent import StopAtTools
from tenacity import retry, stop_after_attempt, wait_fixed, AsyncRetrying
from tqdm import tqdm
from termcolor import colored
import argparse

# ========== Modules ========== #
from .configs.agent_config import *
from .prompts import *
from .conv import run_simulation
from .mes_collect import EvaluationOutputCollector as OutputCollector
from .dataset.mimic_dataset import MIMIC_Dataset
from .tools import TOOLS
from .tools.tool_request import PatientContext

# ========== Logging ========== #
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

set_tracing_disabled(True)

# ========== Config ========== #
@dataclass
class SimulationConfig:
    save_dir: str
    ds_name: str
    ds: MIMIC_Dataset
    medical_assistant_model: str
    medical_assistant_temperature: float
    patient_assistant_model: str
    patient_assistant_temperature: float
    medical_system_prompt: str
    completion_prompt: str
    patient_system_prompt: str
    max_samples: Optional[int] = None
    max_workers: int = 4
    selected_hadm_ids: Optional[List[int]] = None


# ========== Core ========== #
class SimulationRunner:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.patient_gen = self.patient_iterator()

        os.environ["OPENAI_API_KEY"] = (
            API_KEY
        )
        os.environ["OPENAI_BASE_URL"] = (
            BASE_URL
        )

    def patient_iterator(self):
        ds = self.config.ds
        hadm_ids = (
            [hid for hid in ds.hadm_ids if hid in self.config.selected_hadm_ids]
            if self.config.selected_hadm_ids else list(ds.hadm_ids)
        )
        if self.config.max_samples:
            hadm_ids = hadm_ids[:self.config.max_samples]

        logger.info(colored(f"🧬 Preparing {len(hadm_ids)} simulations for {self.config.ds_name} ...", "red"))
        for hadm_id in hadm_ids:
            yield self.prepare_patient(hadm_id)

    def prepare_patient(self, hadm_id: int):
        patient_data = self.config.ds[hadm_id]
        primary_symptom = get_admission_chief_complaint(patient_data)
        medication = "Medication:\n" + get_admission_medication(patient_data)
        anamnesis = patient_data.history_pe_admedication_diagnosis["extracted_history"].values[0] + medication
        hadm_id_str = str(patient_data.admissions.hadm_id.values[0])
        print('hadm_id_str:', hadm_id_str)
        subject_id = str(patient_data.patients.subject_id.values[0])

        collector = OutputCollector(hadm_id=hadm_id_str, dataset_name=self.config.ds_name, save_dir=self.config.save_dir)

        context = PatientContext(
            patient_id=subject_id,
            patient_hadm_id=hadm_id_str,
            patient_data=patient_data,
        )

        med_assistant = Agent[PatientContext](
            name="Doctor",
            instructions=self.config.medical_system_prompt,
            tools=TOOLS,
            # tool_use_behavior=StopAtTools(stop_at_tool_names=["finish"]),
            model=self.config.medical_assistant_model,
            model_settings=ModelSettings(
                temperature=self.config.medical_assistant_temperature,
                tool_choice="auto",
                parallel_tool_calls=False,
                max_tokens=8192,
            ),
        )

        patient_assistant = Agent[PatientContext](
            name="Patient",
            instructions=self.config.patient_system_prompt.format(
                primary_symptom=primary_symptom, anamnesis_summary=anamnesis),
            model=self.config.patient_assistant_model,
            model_settings=ModelSettings(temperature=self.config.patient_assistant_temperature, max_tokens=2048),
        )

        return med_assistant, patient_assistant, primary_symptom, context, collector

    async def run_async(self):
        logger.info(f"🚀 Using asyncio for concurrency")

        semaphore = asyncio.Semaphore(self.config.max_workers)

        tasks = []
        for data in self.patient_gen:
            task = asyncio.create_task(self.run_patient_task_with_limiter(data, semaphore))
            tasks.append(task)

        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Running Simulations"):
            hadm_id, error = await future
            print('hadm_id:', hadm_id)
            if error:
                logger.error(f"❌ Failed HADM_ID {hadm_id}: {error}")
            else:
                logger.info(f"✅ Completed HADM_ID {hadm_id}")

    async def run_patient_task_with_limiter(self, data, semaphore):
        hadm_id = get_hadm_id_safe(data)
        async with semaphore:
            # try:
            async for attempt in AsyncRetrying(stop=stop_after_attempt(3), wait=wait_fixed(1)):
                with attempt:
                    await self.run_patient_task(data)
            return hadm_id, None
            # except Exception as e:
            #     return hadm_id, e

    async def run_patient_task(self, data):
        medical_assistant, patient_assistant, primary_symptom, context, collector = data
        await run_simulation(
            medical_assistant, 
            patient_assistant, 
            primary_symptom, 
            context, 
            output_collector=collector,
            completion_prompt=self.config.completion_prompt)


# ========== 工具函数 ========== #
def get_admission_medication(patient):
    try:
        meds = patient.history_pe_admedication_diagnosis["admission_medication"]
        return meds.values[0] if len(meds.values) > 0 and meds.values[0] else "No current medication."
    except Exception:
        return "No current medication."


def get_admission_chief_complaint(patient):
    cc = patient.triage.chiefcomplaint
    return f"primary symptom(s): {cc.values[0]}" if len(cc.values) > 0 else ""


def get_hadm_id_safe(data_tuple: Tuple) -> Optional[int]:
    try:
        print('data_tuple[2].admissions.hadm_id:', data_tuple[2].admissions.hadm_id)
        return data_tuple[2].admissions.hadm_id.values[0]
    except Exception:
        return None


# ========== 主入口 ========== #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a medical simulation batch with a specific run ID.")
    parser.add_argument('--run_id', type=int, required=True, help='A unique integer ID for this simulation run (e.g., 1, 2, 3...).')
    args = parser.parse_args()

    run_id = args.run_id

    async def main():
        base_output_dir = OUTPUT_DIR
        
        run_specific_base_dir = os.path.join(base_output_dir, f"run_{run_id}")
        
        print(f"==============================================")
        print(f"STARTING SIMULATION BATCH | RUN ID: {run_id}")
        print(f"Output will be saved under: {run_specific_base_dir}")
        print(f"==============================================")

        for dataset_name in DATASET_NAMES:
            original_ids = AIDOC_HADM_IDS[dataset_name]
            #  .../formatted_outputs/run_1/pneumonia
            output_dir_for_dataset = os.path.join(run_specific_base_dir, dataset_name)
            os.makedirs(output_dir_for_dataset, exist_ok=True)

            existing_jsons = glob.glob(os.path.join(output_dir_for_dataset, f"{dataset_name}_*_conversation_*.json"))
            pattern = re.compile(rf"{dataset_name}_(\d+)_conversation_\d+\.json")
            processed_ids = {
                int(match.group(1)) for f in existing_jsons if (match := pattern.match(os.path.basename(f)))
            }

            remaining_ids = [hid for hid in original_ids if hid not in processed_ids]
            log_message = f"🔍 {dataset_name} (Run ID: {run_id}) | Total: {len(original_ids)} | Done: {len(processed_ids)} | Remaining: {len(remaining_ids)}"
            logger.info(log_message) 

            config = SimulationConfig(
                save_dir=output_dir_for_dataset,
                ds_name=dataset_name,
                ds=MIMIC_Dataset.load_dataset(dataset_name),
                medical_assistant_model=MEDICAL_ASSISTANT_MODEL,
                medical_assistant_temperature=MEDICAL_ASSISTANT_TEMPERATURE,
                patient_assistant_model=PATIENT_ASSISTANT_MODEL,
                patient_assistant_temperature=PATIENT_ASSISTANT_TEMPERATURE,
                medical_system_prompt=MEDICAL_SYSTEM_PROMPT,
                completion_prompt=COMPLETION_PROMPT,
                patient_system_prompt=PATIENT_SYSTEM_PROMPT,
                max_samples=MAX_SAMPLES,
                max_workers=MAX_EVAL_PARALLEL_SIZE,
                selected_hadm_ids=remaining_ids,
            )

            await SimulationRunner(config).run_async()
    asyncio.run(main())
