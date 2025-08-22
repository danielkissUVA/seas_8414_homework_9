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

import math

def get_entropy(s):
    """
    Calculates the Shannon entropy of a string.
    This function must be the same as the one used in the training script.
    """
    # Initialize a dictionary 'p' to store character frequencies and get the total length of the string as a float 'lns'.
    # Using a float for 'lns' ensures that division later will result in a float (probability).
    p, lns = {}, float(len(s))

    # Iterate over each character 'c' in the input string 's'.
    for c in s:
        # For each character, get its current count from the dictionary 'p'.
        # If the character is not yet a key in 'p', p.get(c, 0) returns the default value 0.
        # Then, increment the count by 1 and update the dictionary.
        p[c] = p.get(c, 0) + 1

    # Calculate the Shannon entropy using the formula: H = -Σ(p_i * log2(p_i))
    # where p_i is the probability of character i.
    return -sum(
        # 'count / lns' is the probability of a character (frequency / total length).
        (count / lns) * math.log(count / lns, 2)
        # Iterate over all the character counts stored as values in the dictionary 'p'.
        for count in p.values()
    )


def generate_playbook(xai_findings):
    """
    Generates an incident response playbook using a generative AI model
    based on the XAI findings.
    """

    # Retrieve the API key from environment variables. This is a security best practice
    # to avoid hardcoding secrets directly in the source code.
    api_key = os.environ.get("GEMINI_API_KEY")

    # Check if the API key was found. If not, print an informative error message
    # explaining how to set the environment variable and then exit the script.
    if not api_key:
        print("---")
        print("🚨 Error: GEMINI_API_KEY environment variable not set.")
        print("To run this script, you need to set your API key.")
        print("\nFor Linux/macOS, use:\n  export GEMINI_API_KEY='YOUR_API_KEY_HERE'")
        print("\nFor Windows (PowerShell), use:\n  $env:GEMINI_API_KEY=\"YOUR_API_KEY_HERE\"")
        print("\nReplace 'YOUR_API_KEY_HERE' with the key you obtained from Google AI Studio.")
        print("---")
        # Exit with a non-zero status code to indicate that an error occurred.
        sys.exit(1)

    # Construct the full URL for the Google Gemini API endpoint, embedding the API key as a query parameter.
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"

    # Create the prompt for the generative model using an f-string.
    # This prompt provides context and clear instructions, embedding the specific
    # XAI findings to guide the model in generating a relevant playbook.
    prompt = f"""
    Based on the following AI model explanation of a security alert, generate a concise, prescriptive incident response playbook. The playbook should be a step-by-step guide for a cybersecurity analyst.

    AI Model Explanation:
    {xai_findings}

    Playbook:
    """

    # Structure the request payload in the JSON format required by the Gemini API.
    # It specifies the user's role and passes the constructed prompt as the text content.
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ]
    }

    # Use a try...except block to gracefully handle potential network errors or bad API responses.
    try:
        # Implement a retry mechanism with exponential backoff to handle API rate limiting (HTTP 429).
        retries = 0
        while retries < 5:  # Attempt the API call up to 5 times.
            # Send the HTTP POST request to the API with the JSON payload.
            response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'})

            # If the status code is 429 (Too Many Requests), wait and retry.
            if response.status_code == 429:
                # Calculate the sleep time, doubling it with each retry (1, 2, 4, 8, 16 seconds).
                sleep_time = 2 ** retries
                print(f"API rate limit exceeded. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
                retries += 1
                continue  # Go to the next iteration of the loop to retry the request.

            # For any other non-2xx status code, raise an HTTPError to be caught by the except block.
            response.raise_for_status()
            # If the request was successful, break out of the retry loop.
            break
        else:
            # This 'else' block runs only if the 'while' loop completes without a 'break'.
            # This means all retry attempts failed.
            print("Failed to get a successful response after multiple retries.")
            return "Failed to generate playbook due to API issues."

        # Parse the JSON response from the API into a Python dictionary.
        result = response.json()

        # Safely access the generated text from the nested response structure.
        # This checks that the 'result' object and its keys exist and are not empty before access.
        if result and 'candidates' in result and result['candidates']:
            # Extract and return the generated text from the first candidate in the response.
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            # Return an error message if the response format is unexpected.
            return "Failed to generate playbook. The AI response was empty or malformed."

    # Catch any network-related exceptions (e.g., DNS failure, connection error).
    except requests.exceptions.RequestException as e:
        return f"An error occurred while calling the generative AI API: {e}"


def main():
    """
    Main function to parse arguments, load the model, and make a prediction.
    """
    # Load environment variables (like API keys) from a file named .env
    # This keeps sensitive information out of the source code.
    load_dotenv()

    # --- Step 1: Parse Command-line Arguments ---
    # Set up the argument parser to handle command-line inputs.
    parser = argparse.ArgumentParser(description="Analyze a domain name for DGA classification.")
    # Define a required argument '--domain' that the user must provide when running the script.
    parser.add_argument("--domain", required=True, help="The domain name to analyze (e.g., example.com).")
    # Parse the arguments provided by the user.
    args = parser.parse_args()

    # Store the domain name from the parsed arguments into a variable.
    domain_to_analyze = args.domain

    # --- Step 2: Initialize H2O and Load the MOJO Model ---
    print("Initializing H2O...")
    try:
        # Start or connect to an H2O cluster.
        h2o.init()

        # Define the path to the pre-trained MOJO (Model Object, Optimized) model file.
        mojo_path = os.path.join("model", "DGA_Leader.zip")
        # Check if the model file actually exists at the specified path.
        if not os.path.exists(mojo_path):
            print(f"Error: MOJO model not found at {mojo_path}.")
            print("Please run the training script (1_train_and_export.py) first.")
            # If the model is not found, shut down the H2O cluster and exit.
            h2o.shutdown()
            return

        print(f"Loading MOJO model from {mojo_path}...")
        # Upload the MOJO from the zip file into the H2O cluster and assign it to the 'model' variable.
        model = h2o.upload_mojo(mojo_path)

    except Exception as e:
        # Catch any errors that occur during H2O initialization or model loading.
        print(f"An error occurred during H2O initialization or model loading: {e}")
        # Ensure H2O is shut down on error before exiting.
        h2o.shutdown()
        return

    # --- Step 3: Feature Engineering ---
    print(f"Analyzing domain: {domain_to_analyze}")
    # Create the features required by the model for the input domain.
    # The model was trained on 'length' and 'entropy'.
    features = {
        'length': [len(domain_to_analyze)],
        'entropy': [get_entropy(domain_to_analyze)]
    }
    # Convert the Python dictionary of features into an H2OFrame, which is the data format H2O uses.
    df = h2o.H2OFrame(features)

    # --- Step 4: Make a Prediction ---
    print("Making a prediction...")
    # Use the loaded model to make a prediction on the H2OFrame containing the features.
    prediction = model.predict(df)
    # Extract the predicted class ('legit' or 'dga') from the prediction result.
    predicted_class = prediction['predict'][0, 0]
    # Extract the probability that the domain is legitimate.
    legit_probability = prediction['legit'][0, 0]
    # Extract the probability that the domain is a DGA.
    dga_probability = prediction['dga'][0, 0]

    # --- Step 5: Print the Results and Generate Explanation ---
    print("\n--- Prediction Results ---")
    print(f"Domain: {domain_to_analyze}")
    print(f"Predicted Class: {predicted_class}")
    # Print probabilities formatted to four decimal places.
    print(f"Probability (Legit): {legit_probability:.4f}")
    print(f"Probability (DGA): {dga_probability:.4f}")
    print("--------------------------")

    # Only generate a detailed explanation and playbook if the model classifies the domain as DGA.
    if predicted_class == 'dga':
        # --- XAI (Explainable AI) Section ---
        print("\n--- Generating XAI Explanation (SHAP) ---")
        try:
            # Use the model to get SHAP contribution scores for the prediction.
            # This explains how much each feature pushed the prediction towards the final outcome.
            shap_contributions = model.predict_contributions(df)

            # Convert the H2OFrame of SHAP values into a more accessible Python dictionary.
            shap_dict = shap_contributions[0].as_data_frame().to_dict('records')[0]

            # Sort the features by the absolute magnitude of their SHAP contribution, in descending order.
            # This identifies which features were most influential in the model's decision.
            # We exclude 'BiasTerm' as it's a baseline value, not a feature.
            top_features = sorted(
                [(k, v) for k, v in shap_dict.items() if k != 'BiasTerm'],
                key=lambda item: abs(item[1]),
                reverse=True
            )

            # Construct a human-readable explanation string based on the SHAP findings.
            xai_findings = (
                f"- Alert: Potential DGA domain detected.\n"
                f"- Domain: '{domain_to_analyze}'\n"
                f"- AI Model Explanation (from SHAP): The model flagged this domain with "
                f"{dga_probability:.1%} confidence. The classification was primarily driven by:\n"
            )

            # Loop through the most influential features and add them to the explanation string.
            for feature, contribution in top_features:
                # Get the actual value of the feature from the input dataframe.
                value = df[feature][0, 0]
                xai_findings += (
                    # Describe the feature's value and its impact on the prediction.
                    f"  - A {'high' if value > 0 else 'low'} '{feature}' value of "
                    f"{value:.2f} (which strongly pushed the prediction towards 'dga').\n"
                )

            print(xai_findings)

            # --- Step 6: The XAI-to-GenAI Bridge ---
            # Use the explanation text as input for a generative AI model to create an action plan.
            print("\n--- Generating Prescriptive Playbook ---")
            playbook = generate_playbook(xai_findings)
            print(playbook)

        except Exception as e:
            # Catch any errors during the explanation or playbook generation steps.
            print(f"An error occurred during SHAP or playbook generation: {e}")

    # --- Step 7: Shut Down H2O ---
    print("\nShutting down H2O...")
    # Disconnect from the H2O cluster and release its resources.
    h2o.shutdown()

if __name__ == "__main__":
    main()
