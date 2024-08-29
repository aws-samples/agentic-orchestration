import os
import boto3
from langchain_aws import BedrockLLM, ChatBedrock, ChatBedrockConverse, BedrockEmbeddings
#from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from botocore.config import Config

## Setup LLMs
def get_llm(model_id: str, aws_region: str='us-west-2',):
    config = Config(
        retries = dict(
            max_attempts = 10,
            total_max_attempts = 25,
        )
    )
    bedrock_client = boto3.client("bedrock-runtime", config=config, region_name=aws_region)

    inference_modifier = {
        "max_tokens": 4096,
        "temperature": 0.01,
        "top_k": 50,
        "top_p": 0.95,
        "stop_sequences": ["\n\n\nHuman"],
    }

    if 'claude-3-5' in model_id:
        inference_modifier = {
            "max_tokens": 4096,
            "temperature": 0.01,
            "top_k": 50,
            "top_p": 0.95,
            "stop_sequences": ["\n\n\nHuman"],
        }
        llm = ChatBedrock(
            model_id=model_id,
            client=bedrock_client,
            model_kwargs=inference_modifier,
            region_name=aws_region,
        ) 
    elif 'claude-3' in model_id or 'mistral' in model_id:
        llm = ChatBedrockConverse(
            model=model_id,
            client=bedrock_client,
            temperature=0.01,
            max_tokens=4096,
            region_name=aws_region,
        )
    elif 'llama3-1' in model_id:
        llm = ChatBedrockConverse(
            model=model_id,
            client=bedrock_client,
            temperature=0.01,
            max_tokens=2048,
            region_name=aws_region,
        )
    else:
        llm = BedrockLLM(
            model_id=model_id,
            client=bedrock_client,
            model_kwargs={"temperature": 0.1, "max_gen_len":4096},
        )  

    return llm

def get_embedding(model_id: str="amazon.titan-embed-text-v2:0", aws_region: str='us-west-2'):
    config = Config(
        retries = dict(
            max_attempts = 10,
            total_max_attempts = 25,
        )
    )
    bedrock_client = boto3.client("bedrock-runtime", config=config, region_name=aws_region)

    return BedrockEmbeddings(client = bedrock_client, region_name=aws_region, model_id=model_id)