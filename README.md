# Blog Description
[Build Generative AI Applications on Amazon Bedrock with the AWS SDK for Python (Boto3)](https://aws.amazon.com/blogs/machine-learning/build-generative-ai-applications-on-amazon-bedrock-with-the-aws-sdk-for-python-boto3/)

# Requirements
### Prerequisites for Using Amazon Bedrock with Boto3

- An **AWS account** that provides access to AWS services, including Amazon Bedrock.
- The **AWS Command Line Interface (AWS CLI)** set up.
- An **AWS Identity and Access Management (IAM) user** set up for the Amazon Bedrock API with appropriate permissions.
- The **IAM user access key and secret key** to configure the AWS CLI and permissions.  
  - Configure using:  
    ```sh
    aws --configure
    ```
- **Access to Foundation Models (FMs)** on Amazon Bedrock.
- The **latest Boto3 library** installed.
- **Python version 3.8 or later** configured with your integrated development environment (IDE).

# Steps/Actions

## Creat a python file
create  python file named main.py or any other name

## Creating virtual environment in python

Use virtual environment to maintain packages across different projects

```bash
python -m venv venv
```

Activate virtual Enviroment

```bash
venv\Script\activate
```

## Installation of Boto3

Installing boto3 on our virtual environment

```bash
pip install boto3
```

## Code

## 1) Using Anthropic Claude
```python
import boto3
import json

# Set up the Amazon Bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

# Define the model ID
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

# Prepare the input prompt
prompt = "Hello, how are you?"

# Create the request payload
payload = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 2048,
    "temperature": 0.9,
    "top_k": 250,
    "top_p": 1,
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]
}

# Invoke the Amazon Bedrock model
response = bedrock_client.invoke_model(
    modelId=model_id,
    body=json.dumps(payload)
)

# Process the response
result = json.loads(response["body"].read())
generated_text = "".join([output["text"] for output in result["content"]])
print(f"Response: {generated_text}")
```

## Run File 
```bash
python [NAME_OF_FILE].py
```

## OUTPUT
```bash
"Response: Hello! As an AI language model, I don't have feelings or emotions, but I'm operating properly and ready to assist you with any questions or tasks you may have. How can I help you today?"
```

## 2) Using Amazon Nova
```python
import boto3
import json

# Set up the Amazon Bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

# Define the model ID
model_id = "amazon.nova-lite-v1:0"

# Prepare the input prompt
prompt = "Hello, What is amazon bedrock?"

# Invoke the Amazon Bedrock model
response = bedrock_client.invoke_model(
    modelId=model_id,
    body=json.dumps({
    "inferenceConfig": {
      "max_new_tokens": 1000
    },
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "text": prompt
          }
        ]
      }
    ]
  })
)

# Process the response
result = json.loads(response["body"].read())
generated_text = "".join([output["text"] for output in result["output"]["message"]["content"]])
print(f"Response: {generated_text}")
```

## Run File 
```bash
python [NAME_OF_FILE].py
```

## OUTPUT
```bash
"Response: Amazon Bedrock is a service offered by Amazon Web Services (AWS) designed to facilitate the development and 
deployment of generative AI models. Here are some key features and functionalities of Amazon Bedrock:

1. **Model Access**: Bedrock provides access to a variety of foundation models from AWS and third-party providers, allowing developers to leverage pre-trained models for tasks such as natural language processing, image generation, and more.

2. **Customization**: Developers can fine-tune these foundation models with their own data to better suit their specific needs and applications.

3. **Scalability**: Bedrock is built on AWS infrastructure, which means it can easily scale to meet the demands of different applications, from small-scale projects to enterprise-level solutions.

4. **Integration**: The service integrates with other AWS services, such as Amazon SageMaker, to provide a seamless experience for building, training, and deploying machine learning models.

5. **Security and Compliance**: Bedrock includes built-in security features and adheres to various compliance standards, ensuring that sensitive data is protected.

6. **Cost Management**: Pricing is designed to be flexible, often based on usage, allowing organizations to manage costs effectively.

Overall, Amazon Bedrock aims to simplify the process of adopting generative AI by providing a robust platform for model development, customization, and deployment.
"
```