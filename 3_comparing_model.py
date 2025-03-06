import boto3
import json
import time

def invoke_bedrock_model(client, model_id, prompt):
    """Invoke an Amazon Bedrock model and return the response and latency."""
    if "anthropic" in model_id:
        # Payload format for Anthropic Claude models
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "temperature": 0.9,
            "top_k": 250,
            "top_p": 1,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]
        }
    else:
        # Payload format for Amazon models (Nova Lite)
        payload = {
            "inferenceConfig": {"max_new_tokens": 500},
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ]
        }

    start_time = time.time()
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(payload)
    )
    latency = time.time() - start_time

    result = json.loads(response["body"].read())

    if "anthropic" in model_id:
        generated_text = "".join([output["text"] for output in result.get("content", [])])
    else:
        generated_text = "".join([output["text"] for output in result["output"]["message"]["content"]])

    return generated_text, latency

def main():
    # Set up the Amazon Bedrock client
    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1"
    )

    # Define the models and prompt
    models = ["amazon.nova-lite-v1:0", "anthropic.claude-3-sonnet-20240229-v1:0"]
    prompt = "Hello, What is Amazon Bedrock?"

    results = {}
    for model_id in models:
        response_text, response_latency = invoke_bedrock_model(bedrock_client, model_id, prompt)
        results[model_id] = {
            "response": response_text,
            "latency": response_latency
        }

    # Print results
    for model_id, result in results.items():
        print(f"Model: {model_id}")
        print(f"Response: {result['response']}")
        print(f"Latency: {result['latency']:.4f} seconds\n")

if __name__ == "__main__":
    main()

# Model: amazon.nova-lite-v1:0
# Response: Amazon Bedrock is a service offered by Amazon Web Services (AWS) that provides a foundation for building, experimenting with, and deploying generative AI models at scale. Here are some key features and aspects of Amazon Bedrock:

# ### Key Features:

# 1. **Foundation Models**:
#    - Amazon Bedrock offers access to a variety of pre-trained foundation models that can be fine-tuned for specific tasks. These models cover a range of applications such as natural language processing, computer vision, and more.

# 2. **Customization**:
#    - Users can fine-tune these foundation models on their own datasets to tailor the models to their specific needs. This helps in achieving better performance on specific tasks.

# 3. **Scalability**:
#    - The service is designed to handle workloads at scale, enabling businesses to deploy generative AI models efficiently across large datasets and high-volume applications.

# 4. **Integration**:
#    - Amazon Bedrock integrates with other AWS services, making it easier to incorporate generative AI capabilities into existing workflows and applications. This includes services like Amazon SageMaker for machine learning, Amazon S3 for storage, and more.

# 5. **Security and Compliance**:
#    - AWS provides robust security and compliance features to ensure that data and models are protected. This includes encryption, access controls, and compliance with various industry standards.

# 6. **Managed Service**:
#    - As a managed service, Amazon Bedrock takes care of the underlying infrastructure, allowing users to focus on building and deploying their AI models without worrying about the complexities of managing the infrastructure.

# ### Use Cases:

# - **Content Generation**: Automatically generating text, images, or other media content.
# - **Customer Support**: Building intelligent chatbots and virtual assistants to improve customer service.
# - **Personalization**: Tailoring recommendations and user experiences based on individual preferences.
# - **Data Analysis**: Automating the analysis of large datasets to uncover insights and trends.

# ### Getting Started:

# To use Amazon Bedrock, you would typically:
# 1. **Set Up an AWS Account**: If you don't already have one, you'll need to create an AWS account.
# 2. **Access Bedrock**: Navigate to the AWS Management Console and find Amazon Bedrock.
# 3. **Choose a Model**: Select a pre-trained model or upload your own dataset to fine-tune a model.
# 4. **Develop and Deploy**: Use the AWS tools and SDKs to develop, test, and deploy your AI models.

# Amazon Bedrock aims to
# Latency: 5.3683 seconds

# Model: anthropic.claude-3-sonnet-20240229-v1:0
# Response: Amazon Bedrock is a fully-managed service offered by Amazon Web Services (AWS) that provides a way to build, manage, and update applications using containers.

# Here are some key features and capabilities of Amazon Bedrock:

# 1. Container Management: Bedrock allows you to package your application code and dependencies into containers, making it easier to deploy and run applications consistently across different environments.

# 2. Automatic Provisioning: Bedrock automatically provisions and manages the underlying compute infrastructure, such as Amazon Elastic 
# Compute Cloud (EC2) instances, required to run your containerized applications.

# 3. Application Deployment: It provides a streamlined way to deploy and update your containerized applications across multiple environments, such as development, testing, and production.

# 4. Service Discovery: Bedrock integrates with AWS Cloud Map, allowing your applications to discover and communicate with each other seamlessly, even as instances are added or removed.

# 5. Load Balancing: It automatically configures and manages Elastic Load Balancing (ELB) to distribute incoming traffic across your application instances.

# 6. Monitoring and Logging: Bedrock integrates with Amazon CloudWatch for monitoring and logging, allowing you to gain insights into your application's performance and troubleshoot issues.

# 7. Scalability: You can configure autoscaling rules to automatically scale your application instances up or down based on demand, ensuring optimal resource utilization and cost-effectiveness.

# Bedrock aims to simplify the process of building, deploying, and managing containerized applications on AWS, allowing developers to focus more on writing application code rather than managing underlying infrastructure.
# Latency: 10.0840 seconds