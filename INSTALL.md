## Getting Started
### Prerequisites

1. Python 3.x

2. H2O library: `pip install h2o`

3. Requests library: `pip install requests`

4. Python-dotenv library: `pip install python-dotenv`

#### Step 1: Set Up Your Gemini API Key

This project uses the Gemini API to generate the response playbook. You need to obtain an API key from Google AI Studio.

1. Go to Google AI Studio and create a new API key.

2. Create a file named `.env` in the project's root directory.

3. Add your API key to the `.env` file in the following format: `GEMINI_API_KEY='YOUR_API_KEY_HERE'`

#### Step 2: Train and Export the Model

Run the training script to create the dataset and export the DGA detection model.

1. `python 1_train_and_export.py`

This will generate a model/ directory containing the exported model files. You will see a DGA_Leader.zip file, which is the MOJO model used for predictions.

#### Step 3: Analyze a Domain and Generate a Playbook

Use the analysis script to classify a domain and, if it's a DGA, generate an incident response playbook.

#### Example 1: Analyzing a Legitimate Domain

1. `python 2_analyze_domain.py --domain google.com`

**Expected Output**: The script will classify the domain as legit and stop, as no playbook is needed.

#### Example 2: Analyzing a DGA Domain

1. `python 2_analyze_domain.py --domain hjdgsf787gshhshshxnx.com`

**Expected Output**: The script will classify the domain as dga, provide a SHAP explanation for the classification, and then use the Gemini API to generate a prescriptive incident response playbook.