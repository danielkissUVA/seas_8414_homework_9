# Filename: 2_analyze_domain.py
# This script uses the exported H2O MOJO model to classify a given
# domain name, generate a SHAP explanation, and create a prescriptive
# incident response plan using a generative AI model.
import sys

import h2o
import argparse
import math
import os
import requests
import time
from dotenv import load_dotenv # New import

def get_entropy(s):
    """
    Calculates the Shannon entropy of a string.
    This function must be the same as the one used in the training script.
    """
    p, lns = {}, float(len(s))
    for c in s:
        p[c] = p.get(c, 0) + 1
    return -sum(count / lns * math.log(count / lns, 2) for count in p.values())


def generate_playbook(xai_findings):
    """
    Generates an incident response playbook using a generative AI model
    based on the XAI findings.
    """

    # API key is provided by the environment
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print("---")
        print("🚨 Error: GEMINI_API_KEY environment variable not set.")
        print("To run this script, you need to set your API key.")
        print("\nFor Linux/macOS, use:\n  export GEMINI_API_KEY='YOUR_API_KEY_HERE'")
        print("\nFor Windows (PowerShell), use:\n  $env:GEMINI_API_KEY=\"YOUR_API_KEY_HERE\"")
        print("\nReplace 'YOUR_API_KEY_HERE' with the key you obtained from Google AI Studio.")
        print("---")
        sys.exit(1)

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"

    prompt = f"""
    Based on the following AI model explanation of a security alert, generate a concise, prescriptive incident response playbook. The playbook should be a step-by-step guide for a cybersecurity analyst.

    AI Model Explanation:
    {xai_findings}

    Playbook:
    """

    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ]
    }

    try:
        # Implement exponential backoff for API calls
        retries = 0
        while retries < 5:
            response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'})
            if response.status_code == 429:
                sleep_time = 2 ** retries
                print(f"API rate limit exceeded. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
                retries += 1
                continue
            response.raise_for_status()  # Raises an HTTPError for bad responses
            break
        else:
            print("Failed to get a successful response after multiple retries.")
            return "Failed to generate playbook due to API issues."

        result = response.json()

        if result and 'candidates' in result and result['candidates']:
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return "Failed to generate playbook. The AI response was empty or malformed."
    except requests.exceptions.RequestException as e:
        return f"An error occurred while calling the generative AI API: {e}"


def main():
    """
    Main function to parse arguments, load the model, and make a prediction.
    """
    # Load environment variables from the .env file
    load_dotenv()

    # --- Step 1: Parse Command-line Arguments ---
    parser = argparse.ArgumentParser(description="Analyze a domain name for DGA classification.")
    parser.add_argument("--domain", required=True, help="The domain name to analyze (e.g., example.com).")
    args = parser.parse_args()

    domain_to_analyze = args.domain

    # --- Step 2: Initialize H2O and Load the MOJO Model ---
    print("Initializing H2O...")
    try:
        h2o.init()

        # Load the MOJO model using the path from the training script.
        mojo_path = os.path.join("model", "DGA_Leader.zip")
        if not os.path.exists(mojo_path):
            print(f"Error: MOJO model not found at {mojo_path}.")
            print("Please run the training script (1_train_and_export.py) first.")
            h2o.shutdown()
            return

        print(f"Loading MOJO model from {mojo_path}...")
        model = h2o.upload_mojo(mojo_path)

    except Exception as e:
        print(f"An error occurred during H2O initialization or model loading: {e}")
        h2o.shutdown()
        return

    # --- Step 3: Feature Engineering ---
    print(f"Analyzing domain: {domain_to_analyze}")
    features = {
        'length': [len(domain_to_analyze)],
        'entropy': [get_entropy(domain_to_analyze)]
    }
    df = h2o.H2OFrame(features)

    # --- Step 4: Make a Prediction ---
    print("Making a prediction...")
    prediction = model.predict(df)
    predicted_class = prediction['predict'][0, 0]
    legit_probability = prediction['legit'][0, 0]
    dga_probability = prediction['dga'][0, 0]

    # --- Step 5: Print the Results and Generate Explanation ---
    print("\n--- Prediction Results ---")
    print(f"Domain: {domain_to_analyze}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Probability (Legit): {legit_probability:.4f}")
    print(f"Probability (DGA): {dga_probability:.4f}")
    print("--------------------------")

    if predicted_class == 'dga':
        # Generate SHAP explanation for the DGA prediction
        print("\n--- Generating XAI Explanation (SHAP) ---")
        try:
            # Get the SHAP contributions for the single data point.
            shap_contributions = model.predict_contributions(df)

            # Convert to a dictionary for easier access
            shap_dict = shap_contributions[0].as_data_frame().to_dict('records')[0]

            # Find the most impactful features. We will ignore the 'BiasTerm'.
            top_features = sorted(
                [(k, v) for k, v in shap_dict.items() if k != 'BiasTerm'],
                key=lambda item: abs(item[1]),
                reverse=True
            )

            xai_findings = (
                f"- Alert: Potential DGA domain detected.\n"
                f"- Domain: '{domain_to_analyze}'\n"
                f"- AI Model Explanation (from SHAP): The model flagged this domain with "
                f"{dga_probability:.1%} confidence. The classification was primarily driven by:\n"
            )

            # Add the top contributing features to the findings text
            for feature, contribution in top_features:
                value = df[feature][0, 0]
                xai_findings += (
                    f"  - A {'high' if value > 0 else 'low'} '{feature}' value of "
                    f"{value:.2f} (which strongly pushed the prediction towards 'dga').\n"
                )

            print(xai_findings)

            # --- Step 6: The XAI-to-GenAI Bridge ---
            print("\n--- Generating Prescriptive Playbook ---")
            playbook = generate_playbook(xai_findings)
            print(playbook)

        except Exception as e:
            print(f"An error occurred during SHAP or playbook generation: {e}")

    # --- Step 7: Shut Down H2O ---
    print("\nShutting down H2O...")
    h2o.shutdown()


if __name__ == "__main__":
    main()
