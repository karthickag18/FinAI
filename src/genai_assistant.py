
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def explain_risk(prediction, risk_score, customer_data):

    prompt = f"""
    You are a financial risk analysis assistant.

    Customer information:
    {customer_data}

    ML Model Prediction: {prediction}
    Risk Score: {risk_score}

    Explain:
    1. What does this prediction mean?
    2. Which financial factors may matter?
    3. What precautions should be considered?

    Use simple English.
    Do not invent missing customer information.
    Do not make final loan approval decisions.
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

if __name__ == "__main__":

    result = explain_risk(
        prediction="High Risk",
        risk_score="75%",
        customer_data={
            "Credit Score": 580,
            "Income": 30000,
            "Loan Amount": 200000
        }
    )

    print(result)