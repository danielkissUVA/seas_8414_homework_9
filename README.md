## AI-Powered DGA Detection & Incident Response
This project demonstrates an end-to-end cybersecurity workflow that combines a machine learning model for detecting Domain Generation Algorithms (DGAs) with a generative AI model for creating a prescriptive incident response playbook.

### Project Goal
The primary goal of this project is to automate the detection and initial response to DGA-based threats. DGAs are used by malware to generate a large number of domain names for command-and-control (C2) communication, making it difficult for security teams to block them all. By training a model to recognize the characteristics of DGA domains (e.g., high entropy, random-looking strings), we can flag them as malicious. When a DGA domain is detected, the system leverages a generative AI model to produce a clear, step-by-step incident response plan, bridging the gap between detection and action.

### Architecture Overview
The project is built on a three-stage pipeline:

**Training & Export** (1_train_and_export.py): This stage focuses on creating, training, and exporting the DGA detection model.

**Data Generation**: A sample dataset of legitimate and DGA domains is synthetically created. Features like domain length and Shannon entropy (a measure of randomness) are calculated for each domain.

**Model Training**: An H2O AutoML model is trained on this dataset to classify domains as either legit or dga. AutoML automates the machine learning pipeline, including algorithm selection and hyperparameter tuning.

**Model Export**: The best-performing model is exported in two formats:

**MOJO** (Model Object, Optimized): A production-ready, highly portable file (DGA_Leader.zip) that can be used for fast predictions without the full H2O environment.

**H2O Native Format**: The full model object for potential further analysis or retraining.

**Analysis & Response** (2_analyze_domain.py): This stage consumes the exported model to analyze a given domain and generate a response plan.

**Model Loading**: The script loads the production-ready MOJO model.

**Prediction**: It takes a domain name as a command-line argument, calculates its features, and uses the model to predict if it's a DGA.

**XAI (Explainable AI)**: If a domain is classified as DGA, the script uses SHAP (SHapley Additive exPlanations) to explain why the model made that decision. This provides critical context for a human analyst.

**AI-to-AI Bridge**: The explanation from the XAI model is passed as a prompt to a generative AI model (Gemini). This is a powerful "bridge" that transforms a technical model output into a human-readable, actionable security playbook.

**Prescriptive Playbook Generation**: The generative AI model takes the XAI findings and generates a prescriptive, step-by-step incident response playbook. This output is designed to be concise and immediately useful for a cybersecurity analyst, enabling a rapid and standardized response to the detected threat.