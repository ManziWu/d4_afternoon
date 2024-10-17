import random
import os
from utils.ai_inference import gpt4o_inference, gpt4o_mini_inference

def count_files_in_directory(directory_path):
    try:
        entries = os.listdir(directory_path)
        files = [entry for entry in entries if os.path.isfile(os.path.join(directory_path, entry))]
        return len(files)
    except FileNotFoundError:
        return 0

NUM_EMAILS = 600 - count_files_in_directory("./emails")
NUM_CONTRACTS = 100 - count_files_in_directory("./contracts")
NUM_BOARD_PAPERS = 50 - count_files_in_directory("./board_papers")

print(f"NUM EMAILS TO CREATE: {NUM_EMAILS}")
print(f"NUM CONTRACTS TO CREATE: {NUM_CONTRACTS}")
print(f"NUM BOARD PAPERS TO CREATE: {NUM_BOARD_PAPERS}")

SYSTEM_PROMPT = """
Your task is to create a synthetic dataset of emails, contracts and board papers.
The purpose of the dataset is to emulate the documents involved in a mergers and acquisitions due diligence environment.
The target of the acquisition (the company being acquired) is called Canvassian Pty Ltd.
The following is high-level information about Canvassian:
<canvassian>
Canvassian is a software company, based in Australia.
The CEO of Canvassian is Jane Wu. Jane has been an inspirational and very successful leader.
The CTO of Canvassian is Edon Mask. Edon is a visionary technologist.
Canvassian has hundreds of employees, including product managers, software engineers, sales people, and admin people (like internal finance and HR).
Canvassian has 5 major clients: Paywise, Alphabear, Bravocat, Charlemont, Deltaforce and Echona.
Canvassian has hundreds of minor clients.
</canvassian>
You must follow these instructions:
<instructions>
Fully complete each document you create.
You must not leave any placeholders like "[name]". Instead, generate a generic name.
Each document must appear as if it is a real document, in final version.
For an email, randomise who the email is from and too.
For a contract, you must only generate one, complete contract.
For board papers, you must only generate one, complete board paper.
</instructions>
"""

def create_documents(num_emails, num_contracts, num_board_papers):
    documents = []

    def weighted_random_choice(types, weights):
        total_weights = sum(weights.values())
        weighted_choices = list(weights.keys())
        weighted_probabilities = [weights[k] / total_weights for k in weighted_choices]
        chosen_document = random.choices(weighted_choices, weights=weighted_probabilities, k=1)[0]
        return chosen_document
    
    def write_document_to_file(document_type, document_content, index):
        directory = f"./{document_type}"
        document_name = gpt4o_mini_inference(
            "You are a document filename generator for corporate documents.",
            f"""Based on the following content, generate a name for the file:
            <content>
            {document_content}
            </content>
            You must follow these instructions:
            <instructions>
            You must not include a file extension in the name.
            You must not provide any explanation, or provide any output other than the file name.
            """
        )
        os.makedirs(directory, exist_ok=True)
        with open(f"{directory}/{document_name}_{index}.txt", "w") as file:
            file.write(document_content)

    for i in range(num_emails):
        email_type = weighted_random_choice(email_types, email_weights)
        document_content = gpt4o_mini_inference(SYSTEM_PROMPT, email_types[email_type])
        write_document_to_file("emails", document_content, i + 1)

    for i in range(num_contracts):
        contract_type = weighted_random_choice(contract_types, contract_weights)
        document_content = gpt4o_inference(SYSTEM_PROMPT, contract_types[contract_type])
        write_document_to_file("contracts", document_content, i + 1)

    for i in range(num_board_papers):
        board_paper_type = weighted_random_choice(board_paper_types, board_paper_weights)
        document_content = gpt4o_inference(SYSTEM_PROMPT, board_paper_types[board_paper_type])
        write_document_to_file("board_papers", document_content, i + 1)

    return documents

email_types = {
    "spam": "Create a generic spam email.",
    "corporate_general": "Create a generic corporate email.",
    "corporate_risk": "Create an email that includes reference to a corporate risk, like a poor product launch, or customer dissatisfaction.",
    "litigation_risk": "Create an email that includes reference to a litigation risk. Some examples of this include: a regulatory investigation, a competitor suing over patent breach, or a customer suing for a defective product.",
    "paywise_benign": "Create an email that includes a benign reference to the client Paywise.",
    "paywise_risk": "Create an email that includes reference to the client Paywise being in financial difficulty.",
    "alphabear_benign":"Create an email that includes a benign reference to the client Alphabear.",
    "alphabear_risk": "Create an email that includes reference to a corporate risk in respect of the client Alphabear.",
    "bravocat_benign":"Create an email that includes a benign reference to the client Bravocat.",
    "bravocat_risk": "Create an email that includes reference to a corporate risk in respect of the client Bravocat.",
    "charlemont_benign": "Create an email that includes a benign reference to the client Charlemont.",
    "charlemont_risk": "Create an email that includes reference to a corporate risk in respect of the client Charlemont.",
    "deltaforce_benign": "Create an email that includes a benign reference to the client Deltaforce.",
    "deltaforce_risk": "Create an email that includes reference to a corporate risk in respect of the client Deltaforce.",
    "echona_benign": "Create an email that includes a benign reference to the client Echona.",
    "echona_risk": "Create an email that includes reference to a corporate risk in respect of the client Echona.",
    "janewu_benign": "Create an email that includes a benign reference to the CEO, Jane Wu.",
    "janewu_leaving": "Create an email that includes a reference to the CEO, Jane Wu, potentially leaving.",
    "janewu_investigation": "Create an email that alludes to the CEO, Jane Wu, potentially being investigated for some type of corporate misconduct.",
    "cto_benign": "Create an email that includes a benign reference to the CTO, Edon Mask.",
    "cto_leaving": "Create an email that includes a reference to the CTO, Edon Mask, potentially leaving.",
    "cto_investigation": "Create an email that alludes to the CTO, Edon Mask, potentially being investigated for some type of corporate misconduct."
}

email_weights = {
    "spam": 10,
    "corporate_general": 80,
    "corporate_risk": 1,
    "litigation_risk": 1,
    "paywise_benign": 15,
    "paywise_risk": 1,
    "alphabear_benign": 2,
    "alphabear_risk": 0,
    "bravocat_benign": 7,
    "bravocat_risk": 0,
    "charlemont_benign": 5,
    "charlemont_risk": 1,
    "deltaforce_benign": 2,
    "deltaforce_risk": 2,
    "echona_benign": 7,
    "echona_risk": 3,
    "janewu_benign": 10,
    "janewu_leaving": 1,
    "janewu_investigation": 1,
    "cto_benign": 3,
    "cto_leaving": 1,
    "cto_investigation": 2
}

contract_types = {
    "generic": "Draft a generic contract relevant to a software company. This contract must not relate to Paywise, Alphabear, Bravocat, Charlemont, Deltaforce, or Echona. It must relate to a generic, hypothetical company.",
    "paywise": "Draft a contract with Paywise.",
    "alphabear": "Draft a contract with Alphabear. Do not include a change in control clause.",
    "alphabear_risk": "Draft a contract with Alphabear. Include a change in control clause.",
    "bravocat": "Draft a contract with Bravocat. Do not include a change in control clause.",
    "bravocat_risk": "Draft a contract with Bravocat. Include a change in control clause.",
    "charlemont": "Draft a contract with Charlemont. Do not include a change in control clause.",
    "charlemont_risk": "Draft a contract with Charlemont. Include a change in control clause.",
    "deltaforce": "Draft a contract with Deltaforce. Do not include a change in control clause.",
    "deltaforce_risk": "Draft a contract with Deltaforce. Include a change in control clause.",
    "echona": "Draft a contract with Echona. Do not include a change in control clause.",
    "echona_risk": "Draft a contract with Echona. Include a change in control clause."
}

contract_weights = {
    "generic": 40,
    "paywise": 10,
    "alphabear": 3,
    "alphabear_risk": 1,
    "bravocat": 3,
    "bravocat_risk": 1,
    "charlemont": 5,
    "charlemont_risk": 3,
    "deltaforce": 3,
    "deltaforce_risk": 1,
    "echona": 3,
    "echona_risk": 1
}

board_paper_types = {
    "corporate_general": "Draft a generic corporate board paper, discussing regular reporting, and potentially the deal.",
    "corporate_risk": "Draft a corporate board paper that discusses one or more corporate risks, like a poor product launch, customer dissatisfaction, or poor risk controls.",
    "litigation_risk": "Draft a corporate board paper that discusses a litigation risk. Some examples of this include: a regulatory investigation, a competitor suing over patent breach, or a customer suing for a defective product.",
    "paywise_benign": "Draft a corporate board paper that includes a benign reference to the client Paywise (among other things).",
    "paywise_risk": "Draft a corporate board paper that includes a reference to the client Paywise potentially being in financial distress (among other things).",
    "janewu_benign": "Draft a corporate board paper that includes a benign reference to the CEO, Jane Wu (among other things).",
    "janewu_leaving": "Draft a corporate board paper that includes a reference to the CEO, Jane Wu, potentially leaving (among other things).",
    "janewu_investigation": "Draft a corporate board paper that includes a reference to the CEO, Jane Wu, potentially being investigated for corporate misconduct (among other things).",
    "cto_benign": "Draft a corporate board paper that includes a benign reference to the CTO Edon Mask (among other things).",
    "cto_leaving": "Draft a corporate board paper that includes a subtle reference to the CTO, Edon Mask, potentially leaving (among other things).",
    "cto_investigation": "Draft a corporate board paper that includes a subtle reference to the CTO, Edon Masj, potentially being investigated for corporate misconduct (among other things)."
}

board_paper_weights = {
    "corporate_general": 15,
    "corporate_risk": 2,
    "litigation_risk": 1,
    "paywise_benign": 5,
    "paywise_risk": 1,
    "janewu_benign": 3,
    "janewu_leaving": 1,
    "janewu_investigation": 0,
    "cto_benign": 5,
    "cto_leaving": 0,
    "cto_investigation": 2
}

create_documents(NUM_EMAILS, NUM_CONTRACTS, NUM_BOARD_PAPERS)