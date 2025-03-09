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
prompt = "Do you think artificial intelligence will ever surpass human intelligence? Why or why not?"

# Invoke the Amazon Bedrock model
response = bedrock_client.invoke_model(
    modelId=model_id,
    body=json.dumps({
    "inferenceConfig": {
      "max_new_tokens": 500
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

# Output
# Response: Amazon Bedrock is a service offered by Amazon Web Services (AWS) designed to facilitate the development and 
# deployment of generative AI models. Here are some key features and functionalities of Amazon Bedrock:

# 1. **Model Access**: Bedrock provides access to a variety of foundation models from AWS and third-party providers, allowing developers to leverage pre-trained models for tasks such as natural language processing, image generation, and more.

# 2. **Customization**: Developers can fine-tune these foundation models with their own data to better suit their specific needs and applications.

# 3. **Scalability**: Bedrock is built on AWS infrastructure, which means it can easily scale to meet the demands of different applications, from small-scale projects to enterprise-level solutions.

# 4. **Integration**: The service integrates with other AWS services, such as Amazon SageMaker, to provide a seamless experience for building, training, and deploying machine learning models.

# 5. **Security and Compliance**: Bedrock includes built-in security features and adheres to various compliance standards, ensuring that sensitive data is protected.

# 6. **Cost Management**: Pricing is designed to be flexible, often based on usage, allowing organizations to manage costs effectively.

# Overall, Amazon Bedrock aims to simplify the process of adopting generative AI by providing a robust platform for model development, customization, and deployment.