import os
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
import csv
import time
import re
from swarms import Agent, GraphWorkflow, Node, NodeType

print(cerebras.cloud.sdk.__version__)

# Load environment variables from .env file
load_dotenv()

# Retrieve the workspace directory from environment variables
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR")
os.environ["WORKSPACE_DIR"] = WORKSPACE_DIR

# Load environment variables
load_dotenv()
api_key = os.getenv("CEREBRAS_API_KEY")
if not api_key:
    raise ValueError("CEREBRAS_API_KEY environment variable is not set.")

    

# Initialize Cerebras client
client = CerebrasClient(api_key=api_key)
print("Environment set up and model initialized successfully!")

# Define initial seed examples for financial insights
seed_examples = [
    "stock market, investment, risk management", "bond, yield, interest rate", "mutual fund, diversification, asset allocation",
    "real estate, property, mortgage", "cryptocurrency, blockchain, bitcoin", "insurance, policy, premium",
    "retirement, 401k, pension", "budget, savings, expenses", "loan, credit, debt",
    "financial planning, advisor, portfolio", "tax, deduction, IRS", "inflation, deflation, economic indicator",
    "GDP, gross domestic product, economic growth", "recession, depression, economic downturn", "Federal Reserve, monetary policy, interest rates",
    "stock, equity, share", "dividend, earnings, profit", "capital, investment, return on investment",
    "hedge fund, private equity, venture capital", "commodity, gold, oil", "forex, currency, exchange rate",
    "derivative, option, futures", "credit score, FICO, credit report", "mortgage, home loan, interest rate",
    "insurance, health insurance, life insurance", "annuity, retirement income, investment", "budgeting, financial planning, savings",
    "debt, credit card, loan", "investment, stock market, mutual fund", "tax planning, tax return, IRS",
    "economic indicator, GDP, inflation", "monetary policy, Federal Reserve, interest rates", "financial advisor, portfolio, investment strategy",
    "real estate investment, property, rental income", "cryptocurrency investment, bitcoin, blockchain", "insurance policy, premium, deductible",
    "retirement planning, 401k, pension", "budgeting, savings, expenses", "loan, credit, debt management",
    "financial planning, advisor, portfolio management", "tax planning, deduction, IRS", "inflation, deflation, economic indicator",
    "GDP, gross domestic product, economic growth", "recession, depression, economic downturn", "Federal Reserve, monetary policy, interest rates",
    "stock, equity, share", "dividend, earnings, profit", "capital, investment, return on investment",
    "hedge fund, private equity, venture capital", "commodity, gold, oil", "forex, currency, exchange rate",
    "derivative, option, futures", "credit score, FICO, credit report", "mortgage, home loan, interest rate",
    "insurance, health insurance, life insurance", "annuity, retirement income, investment", "budgeting, financial planning, savings",
    "debt, credit card, loan", "investment, stock market, mutual fund", "tax planning, tax return, IRS",
    "economic indicator, GDP, inflation", "monetary policy, Federal Reserve, interest rates", "financial advisor, portfolio, investment strategy",
    "real estate investment, property, rental income", "cryptocurrency investment, bitcoin, blockchain", "insurance policy, premium, deductible",
    "retirement planning, 401k, pension", "budgeting, savings, expenses", "loan, credit, debt management"
]

# Define the function to initialize the agent
def create_agent_response(prompt):
    max_retries = 5
    retry_delay = 60  # seconds

    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "system", "content": "You are a knowledgeable financial advisor."},
                          {"role": "user", "content": prompt}],
                model="llama3.1-8b",
            )
            response = chat_completion.choices[0].message.content
            print(f"Response received: {response[:100]}...")  # Print first 100 characters of the response for debug
            return response
        except cerebras_cloud.InternalServerError as e:
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    raise Exception("Max retries exceeded. Service is unavailable.")

# Initialize financial advisor agent
financial_advisor = Agent(
    agent_name="FinancialAdvisor",
    llm=create_agent_response,
    max_loops=1,
    autosave=True,
    dashboard=False,
    system_prompt=(
        "You are an expert financial advisor. For each financial term or scenario provided, generate a financial insight or recommendation. "
        "Break down the insight into individual components or steps. "
        "Ensure that each component is clearly explained and enclose the financial term, explanation, and recommendation in brackets like this: '[Financial Term] [Explanation] [Recommendation]'. "
        "Use Markdown formatting and ensure each insight is accurate."
    )
)

# Setup workflow graph
wf_graph = GraphWorkflow()
wf_graph.add_node(Node(id="financial_advisor", type=NodeType.AGENT, agent=financial_advisor))
print("Workflow graph created successfully!")

# Generate a prompt for the agent
def generate_financial_prompt(subject):
    full_prompt = (
        f"Generate a financial insight or recommendation based on the following financial term or scenario: {subject}. "
        f"For each component or step, include the financial term, explanation, and recommendation. "
        f"Ensure that each component is clearly explained, and enclose the insights in brackets as follows: '[Financial Term] [Explanation] [Recommendation]'. "
    )
    try:
        return create_agent_response(full_prompt)
    except Exception as e:
        print(f"Error generating financial insight for subject '{subject}': {e}")
        return None

# Clean and format output
def clean_output(output):
    matches = re.findall(r'\[(.*?)\]', output)
    return ', '.join(matches)

# Save cleaned output to CSV
def save_to_csv(csv_filename, output_text):
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([output_text])

def save_to_txt(txt_filename, output_text):
    with open(txt_filename, mode='a', encoding='utf-8') as file:
        file.write(output_text + '\n')

# Add new cleaned output to seed examples if not already present
def add_to_seed_bank(cleaned_output, seed_examples):
    if cleaned_output not in seed_examples:
        seed_examples.append(cleaned_output)
        print(f"Added new example to seed bank: {cleaned_output}")
    else:
        print(f"Example already exists in seed bank: {cleaned_output}")

# Main processing loop
def main_loop(seed_examples, csv_filename, txt_filename):
    used_seeds = set()

    while seed_examples:
        seed = seed_examples.pop(0)
        if seed in used_seeds:
            print(f"Skipping used seed: {seed}")
            continue

        try:
            output = generate_financial_prompt(seed)
            if output:
                cleaned_output = clean_output(output)
                save_to_csv(csv_filename, cleaned_output)
                save_to_txt(txt_filename, cleaned_output)
                add_to_seed_bank(cleaned_output, seed_examples)
                used_seeds.add(seed)
                print(f"Seed '{seed}' processed and saved.")
        except Exception as e:
            print(f"Error processing seed '{seed}': {e}")

    print("All seed examples have been processed.")

# Run the main loop
if __name__ == "__main__":
    csv_file = 'financial_insights.csv'
    txt_file = 'financial_insights.txt'
    main_loop(seed_examples, csv_file, txt_file)
