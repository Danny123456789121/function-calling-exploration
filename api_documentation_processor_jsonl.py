import os
import json
from collections import defaultdict
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client

def process_file(file_path, chain):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = json.load(file)
        
        content_str = json.dumps(content, indent=2)
        print(f"Content of {file_path}:\n{content_str[:300]}...")
        
        func_name = content.get('name', 'Unknown Function')
        func_desc = content_str  

        prompt = f"""
            You are a data labeler. The responsibility for you is to
            generate a set of diverse queries and corresponding
            answers for the given functions in JSON format.
            Construct queries and answers that exemplify how to use
            these functions in a practical scenario. Include in each
            query specific, plausible values for each parameter. For
            instance, if the function requires a date, use a typical
            and reasonable date.

            Ensure the query:
            − Is clear and concise
            − Contains multiple parallel queries in natural language for
            the given functions, they could use either the same
            function with different params or different functions
            − Demonstrates typical use cases
            − Includes all necessary parameters in a meaningful way. For
            numerical parameters, it could be either numerals or words
            − Covers a variety of difficulty levels, ranging from
            beginner to advanced use cases
            − The corresponding result's parameter types and ranges match
            with the functions descriptions

            Ensure the answer:
            − Is a list of function calls in JSON format
            − The length of the answer list should be equal to the number
            of requests in the query
            − Can solve all the requests in the query effectively
            − Includes detailed API information including name, URL, method, endpoint, and parameters

            Important:
            − Generate at least 10 diverse query and answer pairs
            − Include a mix of single API calls and batch requests (multiple API calls in one query)
            − Ensure that at least 30% of the queries are batch requests
            − Create as many unique query structures as possible, varying the combination of API calls and their parameters

            Based on these instructions, generate diverse query and answer pairs for the function '{func_name}'.
            The detailed function description is as follows:
            {func_desc}

            The output MUST strictly adhere to the following JSON format,
            and NO other text MUST be included:
            [
            {{
            "query": "The generated query.",
            "answers": [
            {{
            "api_name": "API_NAME",
            "url": "https://{{host}}:{{port}}/example/url",
            "method": "HTTP_METHOD",
            "endpoint": "/example/endpoint",
            "params": {{
            "arg_name": "value",
            ... (more params as required)
            }}
            }},
            ... (more API calls as required)
            ]
            }}
            ]
            """
        
        response = chain.invoke(prompt)
        
        try:
            result = json.loads(response['text'])
            
            for entry in result:
                entry['file'] = os.path.basename(file_path)
                
            return result
        except json.JSONDecodeError:
            print(f"Invalid JSON in response for file {file_path}")
            return None
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def analyze_queries(jsonl_data):
    unique_structures = defaultdict(int)
    batch_requests = defaultdict(int)
    single_requests = 0

    for entry in jsonl_data:
        structure_key = tuple(
            (
                answer['api_name'],
                answer['url'],
                answer['method'],
                answer['endpoint'],
                tuple(answer.get('params', {}).keys() if isinstance(answer.get('params', {}), dict) else range(len(answer.get('params', []))))
            ) for answer in entry['answers']
        )
        unique_structures[structure_key] += 1

        num_calls = len(entry['answers'])
        if (num_calls > 1):
            batch_requests[num_calls] += 1
        else:
            single_requests += 1

    total_queries = len(jsonl_data)
    unique_query_count = len(unique_structures)
    duplicate_query_count = total_queries - unique_query_count

    return {
        'total_queries': total_queries,
        'unique_query_structures': unique_query_count,
        'duplicate_query_structures': duplicate_query_count,
        'batch_requests': dict(batch_requests),
        'single_requests': single_requests
    }

def generate_summary(analysis):
    batch_request_summary = ", ".join([f"{count} requests with {calls} calls" for calls, count in analysis['batch_requests'].items()])
    total_batch_requests = sum(analysis['batch_requests'].values())
    
    summary = f"""
Summary:
- Total queries: {analysis['total_queries']}
- Unique query structures: {analysis['unique_query_structures']}
- Duplicate query structures: {analysis['duplicate_query_structures']}
- Batch requests (multiple API calls in one query): {total_batch_requests}
  Breakdown: {batch_request_summary}
- Single requests (one API call per query): {analysis['single_requests']}

This analysis shows that out of {analysis['total_queries']} total queries, there are {analysis['unique_query_structures']} unique query structures and {analysis['duplicate_query_structures']} duplicates. {total_batch_requests} queries are batch requests (bundling multiple API calls), with a detailed breakdown provided above. {analysis['single_requests']} are single requests.
    """
    return summary

def main():
    print("Starting main function")
    
    proxy_client = get_proxy_client('gen-ai-hub')
    chat_llm = ChatOpenAI(proxy_model_name='gpt-4', proxy_client=proxy_client)
    
    system_template = "You are an expert API documentation analyst. Your task is to analyze API documentation and generate queries and example API calls based on the given prompt."
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    
    chain = LLMChain(llm=chat_llm, prompt=chat_prompt)
    
    folder_path = r".\APIs"
    print(f"Looking for files in: {folder_path}")
    
    if not os.path.exists(folder_path):
        print(f"Error: The directory {folder_path} does not exist.")
        return
    
    all_files = os.listdir(folder_path)
    print(f"All files in the directory: {all_files}")
    
    json_files = [f for f in all_files if f.endswith('.json')]
    if not json_files:
        print("No .json files found in the directory.")
        return
    
    jsonl_data = []
    
    for filename in json_files:
        file_path = os.path.join(folder_path, filename)
        print(f"Processing file: {filename}")
        
        result = process_file(file_path, chain)
        
        if result:
            jsonl_data.extend(result)
            
            print(f"Completed processing {filename}")
            print(f"Generated result: {json.dumps(result, indent=2)[:300]}...")
        else:
            print(f"Failed to process {filename}")
        
        print("------------------------")
    
    jsonl_file_path = os.path.join(folder_path, "output_dataset.jsonl")
    with open(jsonl_file_path, "w", encoding='utf-8') as jsonl_file:
        for entry in jsonl_data:
            jsonl_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"Results saved to {jsonl_file_path}")
    
    analysis = analyze_queries(jsonl_data)
    summary = generate_summary(analysis)
    print(summary)

if __name__ == "__main__":
    main()
