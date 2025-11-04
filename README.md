Step 1. Create a new env and install related packages. 

Step 2. 
   ```sh
   cd Hospital_Agent 
   ```

Step 3. check the dataset source in Hospital_Agent/src/dataset/mimic_dataset.py, check the general config in Hospital_Agent/src/configs/agent_config.py

Step 4. add .env for OPENAI_BASE_URL and OPENAI_API_KEY under Hospital_Agent

Step 5. 
   ```sh
   bash src/run_simulations.sh 
   ```