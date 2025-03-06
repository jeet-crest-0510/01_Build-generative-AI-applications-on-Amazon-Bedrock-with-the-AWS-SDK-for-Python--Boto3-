import boto3
import json
import time

def invoke_bedrock_model(client, model_id, prompt):
    """Invoke an Amazon Bedrock model and return the response and latency."""
    if "anthropic" in model_id:
        # Payload format for Anthropic Claude models
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,
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
            "inferenceConfig": {"max_new_tokens": 100},
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

    # Define the models and prompts
    models = ["amazon.nova-lite-v1:0", "anthropic.claude-3-sonnet-20240229-v1:0"]
    prompts = [
        "Hello, What is Amazon Bedrock?",
        "What are the key differences between classical physics and quantum physics?",
        "If a farmer has 20 chickens and he buys 15 more, but 5 escape, how many chickens does he have left? Explain your reasoning.",
        "Explain how blockchain technology ensures security and decentralization in cryptocurrencies.",
        "Do you think artificial intelligence will ever surpass human intelligence? Why or why not?"
    ]

    results = {}
    for prompt in prompts:
        print(f"Prompt: {prompt}\n")
        for model_id in models:
            response_text, response_latency = invoke_bedrock_model(bedrock_client, model_id, prompt)
            results[(model_id, prompt)] = {
                "response": response_text,
                "latency": response_latency
            }
            print(f"Model: {model_id}")
            print(f"Response: {response_text}")
            print(f"Latency: {response_latency:.4f} seconds\n")
        print("-" * 80)

if __name__ == "__main__":
    main()


# Output

# Prompt: Hello, What is Amazon Bedrock?

# Model: amazon.nova-lite-v1:0
# Response: Amazon Bedrock is a service provided by Amazon Web Services (AWS) designed to simplify the development of generative AI applications. It serves as a foundation to build, test, and scale machine learning models, particularly those involving generative AI, which can create content such as text, images, and more.

# Here are some key features and components of Amazon Bedrock:

# 1. **Foundation Models**: Bedrock offers access to a variety of pre-trained foundation models, which are large-scale AI models trained on
# Latency: 3.3069 seconds

# Model: anthropic.claude-3-sonnet-20240229-v1:0
# Response: Amazon Bedrock is a service offered by Amazon Web Services (AWS) that provides a managed service for running computing fabrics on premises.

# Some key points about Amazon Bedrock:

# - It allows organizations to run compute, storage, database and other cloud services on-premises in their own data centers using the same APIs and control plane as the AWS cloud.

# - It provides the same APIs, tools, and functionality as AWS cloud services, but deployed locally behind a company's firewall
# Latency: 4.7415 seconds

# --------------------------------------------------------------------------------
# Prompt: What are the key differences between classical physics and quantum physics?

# Model: amazon.nova-lite-v1:0
# Response: Classical physics and quantum physics represent two fundamental frameworks for understanding the universe, each with its own principles, scope, and implications. Here are the key differences between the two:

# ### 1. **Scale of Application:**
# - **Classical Physics:** Applies well to macroscopic objects (larger than atoms and molecules), such as planets, stars, and everyday objects.
# - **Quantum Physics:** Primarily deals with microscopic objects (atoms, subatomic particles), although it can also describe macroscopic phenomena under certain conditions
# Latency: 1.2410 seconds

# Model: anthropic.claude-3-sonnet-20240229-v1:0
# Response: The key differences between classical physics and quantum physics are:

# 1. Scale: Classical physics applies to the macroscopic world of everyday objects, while quantum physics deals with the microscopic world of atoms, molecules, and subatomic particles.

# 2. Determinism vs. Probability: Classical physics is deterministic, meaning that if you know the initial conditions of a system, you can predict its future state with certainty. Quantum physics, on the other hand, is probabilistic      
# Latency: 2.7341 seconds

# --------------------------------------------------------------------------------
# Prompt: If a farmer has 20 chickens and he buys 15 more, but 5 escape, how many chickens does he have left? Explain your reasoning.

# Model: amazon.nova-lite-v1:0
# Response: To determine how many chickens the farmer has left, we can break down the problem into a series of simple steps:

# 1. **Initial Number of Chickens:**
#    The farmer starts with 20 chickens.

# 2. **Chickens Bought:**
#    The farmer buys 15 more chickens.
#    \[
#    20 \text{ (initial chickens)} + 15 \text{ (bought chickens)} = 35 \text{ chickens}
#    \]

# 3. **
# Latency: 1.0996 seconds

# Model: anthropic.claude-3-sonnet-20240229-v1:0
# Response: To solve this problem, we need to follow the given information and perform the necessary calculations.

# Given information:
# - The farmer initially has 20 chickens.
# - The farmer buys 15 more chickens.
# - 5 chickens escape.

# Step 1: Find the total number of chickens after buying 15 more.
# Initial number of chickens + Chickens bought = Total number of chickens
# 20 + 15 =
# Latency: 2.7467 seconds

# --------------------------------------------------------------------------------
# Prompt: Explain how blockchain technology ensures security and decentralization in cryptocurrencies.

# Model: amazon.nova-lite-v1:0
# Response: Blockchain technology ensures security and decentralization in cryptocurrencies through several key mechanisms:

# ### 1. Decentralization:
# - **Distributed Ledger**: Unlike traditional centralized systems where a single entity controls the data, blockchain utilizes a distributed ledger. This ledger is maintained by a network of nodes (computers) spread across the globe.    
# - **Consensus Mechanisms**: To add a new block to the blockchain, consensus mechanisms like Proof of Work (PoW), Proof of Stake (PoS), or Delegate
# Latency: 1.0863 seconds

# Model: anthropic.claude-3-sonnet-20240229-v1:0
# Response: Blockchain technology ensures security and decentralization in cryptocurrencies through several key mechanisms:

# 1. Decentralization: In a blockchain network, there is no central authority or server controlling the system. Instead, the network is distributed across a vast number of nodes (computers) globally. Each node maintains a copy of the entire blockchain ledger, making it highly resistant to censorship, manipulation, or shutdown by any single entity.       

# 2. Cryptographic Hashing: Each block
# Latency: 2.6685 seconds

# --------------------------------------------------------------------------------
# Prompt: Do you think artificial intelligence will ever surpass human intelligence? Why or why not?

# Model: amazon.nova-lite-v1:0
# Response: The question of whether artificial intelligence (AI) will ever surpass human intelligence is a complex and multifaceted one, with perspectives ranging from optimism to caution. Here are some key points to consider:

# ### Reasons Why AI Might Surpass Human Intelligence:

# 1. **Computational Power**:
#    - **Exponential Growth**: Moore's Law suggests that the number of transistors on a microchip doubles approximately 
# every two years, leading to exponential increases in computational power.
#    - **Parallel Processing**: AI systems can process vast
# Latency: 1.1264 seconds

# Model: anthropic.claude-3-sonnet-20240229-v1:0
# Response: I don't have a definitive answer on whether AI will surpass human intelligence, as there are many complex factors involved. However, here are some points to consider:

# - AI systems are already superior to humans in certain narrow domains that require rapid processing of huge amounts of data, like playing chess or doing complex calculations. As AI continues advancing, the domains where it exceeds human capabilities will likely expand.

# - However, human intelligence is remarkably flexible, general, and driven by high-
# Latency: 3.8090 seconds

# --------------------------------------------------------------------------------




# Prompt: Hello, What is Amazon Bedrock?

# Model: amazon.nova-lite-v1:0
# Response: Amazon Bedrock is a service offered by Amazon Web Services (AWS) designed to simplify the development and deployment of generative AI applications. It provides a foundation for building, experimenting, and scaling AI models. Here are some key features and components of Amazon Bedrock:

# ### Key Features:
# 1. **Model Access**:
#    - **Pre-trained Models**: Access to a variety of pre-trained generative AI models from AWS and third-party providers.
#    - **Custom Models**: Ability to bring your own pre-trained models or fine-tune models using your data.

# 2. **Infrastructure Management**:
#    - **Scalability**: Automatically scales the infrastructure to handle varying workloads.
#    - **Cost Efficiency**: Pay-as-you-go pricing model to optimize costs based on usage.

# 3. **Model Customization**:
#    - **Fine-Tuning**: Customize pre-trained models with your own data to better fit your specific use case.
#    - **Model Evaluation**: Tools and metrics to evaluate the performance of your models.

# 4. **Security and Compliance**:
#    - **Data Security**: Built-in data encryption and compliance with various industry standards.
#    - **Access Control**: Fine-grained access controls to manage who can use and manage the models.

# 5. **Integration**:
#    - **APIs and SDKs**: Easy integration with other AWS services and third-party applications using APIs and SDKs.    
#    - **Developer Tools**: Tools and templates to facilitate rapid development and deployment.

# ### Use Cases:
# - **Content Generation**: Automatically generating text, images, and other content.
# - **Chatbots**: Building intelligent chatbots for customer service and other applications.
# - **Data Analysis**: Generating insights and summaries from large datasets.
# - **Personalization**: Tailoring user experiences based on individual preferences and behaviors.

# ### How It Works:
# 1. **Model Selection**: Choose from a range of pre-trained models or bring your own.
# 2. **Deployment**: Deploy the model using Amazon Bedrock’s managed infrastructure.
# 3. **Customization**: Fine-tune the model with your data to improve its performance for your specific use case.       
# 4. **Monitoring and Management**: Use built-in tools to monitor performance and manage the model lifecycle.

# Amazon Bedrock aims to lower the barriers to entry for businesses and developers looking to leverage generative AI by 
# providing a robust, scalable, and secure platform.
# Latency: 5.3666 seconds

# Model: anthropic.claude-3-sonnet-20240229-v1:0
# Response: Amazon Bedrock is a real-time operating system developed by Amazon for running applications on resource-constrained devices and embedded systems.

# Some key points about Amazon Bedrock:

# - It is a secure, real-time operating system intended for use in Internet of Things (IoT) devices, industrial equipment, consumer appliances, and other embedded systems.

# - Bedrock is based on the open-source FreeRTOS kernel and provides additional components and libraries for building connected device applications.

# - It aims to provide a secure foundation for IoT devices by incorporating security features like code signing, encrypted communications, and secure boot capabilities.

# - Bedrock supports microcontroller units (MCUs) and applications processors from various chip manufacturers.

# - It integrates with AWS IoT services, allowing devices to securely connect to the AWS cloud for data exchange, updates, and management.

# - Amazon provides Bedrock for free, including the source code, to enable customization for different hardware platforms and use cases.

# The goal of Bedrock is to simplify the development of secure, connected device software by providing a real-time OS with pre-integrated security capabilities and seamless integration with AWS cloud services.
# Latency: 7.9384 seconds

# --------------------------------------------------------------------------------
# Prompt: How does Amazon Bedrock compare to other AI services?

# Model: amazon.nova-lite-v1:0
# Response: Amazon Bedrock is a managed service from Amazon Web Services (AWS) that aims to simplify the process of building, sharing, and scaling generative AI applications. To understand how it compares to other AI services, we can look at several key aspects:

# ### 1. **Ease of Use**
# - **Amazon Bedrock**: Designed to be user-friendly, with a focus on making it easy for developers to build and deploy 
# generative AI models. It provides a streamlined interface and pre-built components.
# - **Other AI Services (e.g., Google AI Platform, Microsoft Azure Machine Learning, IBM Watson)**: These platforms also offer extensive tools and SDKs, but the learning curve might be steeper, particularly for those new to the platform. 

# ### 2. **Model Support and Customization**
# - **Amazon Bedrock**: Provides a range of pre-trained models that can be fine-tuned for specific tasks. It supports various model architectures and allows for customization to fit specific needs.
# - **Other AI Services**: Most major cloud providers offer a variety of pre-trained models and support custom model training. For example, Google's AI Platform offers TensorFlow and PyTorch support, while Azure has Azure Machine Learning and Cognitive Services.

# ### 3. **Integration with AWS Ecosystem**
# - **Amazon Bedrock**: Seamlessly integrates with other AWS services like Amazon S3, Amazon SageMaker, and AWS Lambda, 
# which can be advantageous for users already embedded in the AWS ecosystem.
# - **Other AI Services**: Integration capabilities vary. Google AI Platform integrates well with Google Cloud Platform 
# services, and Azure Machine Learning integrates with Microsoft Azure services.

# ### 4. **Scalability and Performance**
# - **Amazon Bedrock**: Designed to scale with your needs, leveraging AWS's robust infrastructure to handle large workloads and high throughput.
# - **Other AI Services**: All major cloud providers offer scalable solutions, but the specifics of how they handle scaling and performance can vary. AWS, Google Cloud, and Azure all have strong track records in this area.

# ### 5. **Cost**
# - **Amazon Bedrock**: Pricing is competitive and follows the typical AWS pay-as-you-go model, but specific pricing details can vary based on usage and model selection.
# - **Other AI Services**: Pricing models are similar, with pay-as-you-go options. Costs can vary based on the specific 
# services used and the volume of requests.

# ### 6. **Community and Support**
# - **Amazon Bed
# Latency: 4.3636 seconds

# Model: anthropic.claude-3-sonnet-20240229-v1:0
# Response: I do not actually have detailed information about Amazon's Bedrock AI service. I am Claude, an AI assistant 
# created by Anthropic to be helpful, harmless, and honest. I don't have insider knowledge of specific products or services offered by companies like Amazon.
# Latency: 2.0009 seconds

# --------------------------------------------------------------------------------
# Prompt: What are the key features of Amazon Bedrock?

# Model: amazon.nova-lite-v1:0
# Response: Amazon Bedrock is a managed service designed to help developers build, share, and scale generative AI applications. Here are some of its key features:

# ### 1. **Foundation Models**
# - **Pre-trained Models**: Access to a variety of pre-trained generative AI models that can be fine-tuned for specific 
# tasks.
# - **Customization**: Ability to customize and fine-tune models with your own data to better suit specific use cases.  

# ### 2. **Model Deployment**
# - **Ease of Use**: Simplifies the deployment of models with streamlined workflows.
# - **Scalability**: Automatically scales to handle large workloads and high traffic demands.

# ### 3. **Integration**
# - **AWS Ecosystem**: Seamless integration with other AWS services such as Amazon SageMaker, AWS Lambda, and Amazon API Gateway.
# - **Third-Party Tools**: Compatibility with various third-party tools and platforms.

# ### 4. **Security and Compliance**
# - **Data Security**: Built-in data protection features to ensure secure data handling.
# - **Compliance**: Adherence to various industry standards and regulations to meet compliance requirements.

# ### 5. **Monitoring and Management**
# - **Performance Monitoring**: Tools for monitoring model performance and making adjustments as needed.
# - **Cost Management**: Features for tracking and managing costs associated with model usage.

# ### 6. **Collaboration**
# - **Model Sharing**: Ability to share models across teams and projects.
# - **Version Control**: Tools for managing different versions of models.

# ### 7. **Developer Tools**
# - **SDKs and APIs**: Support for various programming languages through SDKs and APIs.
# - **Jupyter Notebooks**: Integration with Jupyter notebooks for interactive model development and experimentation.    

# ### 8. **Real-Time Inference**
# - **Low Latency**: Optimized for low-latency inference to support real-time applications.
# - **Batch Processing**: Support for batch processing tasks in addition to real-time inference.

# ### 9. **Documentation and Support**
# - **Comprehensive Documentation**: Detailed documentation and guides to help developers get started.
# - **Community and Support**: Access to a community of developers and technical support from AWS.

# ### 10. **Experimentation and Iteration**
# - **A/B Testing**: Tools for conducting A/B testing to compare model performance.
# - **Feedback Loops**: Mechanisms for incorporating user feedback into model improvements.

# These features make Amazon Bedrock a powerful tool for developers looking to leverage generative AI in
# Latency: 4.5067 seconds

# Model: anthropic.claude-3-sonnet-20240229-v1:0
# Response: Amazon Bedrock is a service from Amazon Web Services (AWS) that provides a foundation for building real-time applications. Here are some key features of Amazon Bedrock:

# 1. Real-time data streaming: Bedrock allows you to ingest, process, and analyze real-time data streams from various sources, such as IoT devices, website clickstreams, and mobile applications.

# 2. Serverless architecture: Bedrock is built on AWS Lambda, which allows you to run code without provisioning or managing servers. This serverless approach makes it easier to build and scale real-time applications.

# 3. Low latency: Bedrock is designed to handle real-time data with low latency, enabling applications to respond to events quickly.

# 4. Data integration: Bedrock integrates with other AWS services like Amazon Kinesis, Amazon DynamoDB, and Amazon S3, allowing you to easily ingest, process, and store data from different sources.

# 5. Scalability: Bedrock automatically scales up or down based on the incoming data volume, ensuring that your application can handle varying workloads without manual intervention.

# 6. Monitoring and debugging: Bedrock provides monitoring and debugging tools to help you understand the performance and behavior of your real-time applications.

# 7. Security: Bedrock leverages AWS's security features, such as identity and access management (IAM) and encryption, to help secure your data and applications.

# Bedrock is particularly useful for building applications that require real-time data processing, such as IoT applications, streaming analytics, and real-time dashboards. It simplifies the development and deployment of these types of applications by providing a serverless, scalable, and low-latency infrastructure.
# Latency: 6.1798 seconds

# --------------------------------------------------------------------------------
# Prompt: How can developers use Amazon Bedrock?

# Model: amazon.nova-lite-v1:0
# Response: Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) from leading AI companies like AI21 Labs, Anthropic, Cohere, and Amazon. It provides a variety of tools and APIs to build, evaluate, and scale generative AI applications. Here’s how developers can use Amazon Bedrock:

# ### 1. Model Selection and Deployment
# - **Choose a Model**: Developers can select from a variety of pre-trained foundation models based on their specific needs, such as text generation, summarization, or code completion.
# - **Customization**: Fine-tune selected models on custom datasets to better suit specific business requirements.      

# ### 2. API Access
# - **Bedrock API**: Use the Bedrock API to interact with the chosen models programmatically. This allows developers to 
# integrate models into their applications easily.
# - **SDKs and Libraries**: Utilize AWS SDKs and libraries available in multiple programming languages (e.g., Python, Java, JavaScript) to streamline the integration process.

# ### 3. Model Evaluation
# - **Evaluation Tools**: Use built-in tools to evaluate model performance on various metrics, ensuring the chosen model meets the required standards for accuracy, speed, and cost.
# - **A/B Testing**: Implement A/B testing frameworks to compare different models or versions of the same model to determine the best fit for the application.

# ### 4. Scalability and Management
# - **Auto-Scaling**: Leverage AWS infrastructure to automatically scale model inference based on demand.
# - **Monitoring and Logging**: Use Amazon CloudWatch to monitor model performance and logs, ensuring optimal operation 
# and quick troubleshooting.

# ### 5. Security and Compliance
# - **IAM Integration**: Use AWS Identity and Access Management (IAM) to control access to Bedrock resources.
# - **Data Protection**: Ensure data protection and compliance with industry standards and regulations by using encryption and other security features provided by AWS.

# ### 6. Integration with Other AWS Services
# - **SageMaker**: Integrate Bedrock with Amazon SageMaker for advanced model training and deployment capabilities.     
# - **Lambda**: Use AWS Lambda to create serverless functions that leverage Bedrock models for real-time processing.    
# - **API Gateway**: Deploy Bedrock models behind an API Gateway for secure and scalable access.

# ### 7. Development and Testing
# - **Local Testing**: Use the AWS CLI and SDKs to test models locally before deploying them to production.
# - **Jupy
# Latency: 4.4432 seconds

# Model: anthropic.claude-3-sonnet-20240229-v1:0
# Response: Amazon Bedrock is not an actual Amazon product or service. Amazon does offer various cloud services and tools for developers, but there is no specific offering called "Amazon Bedrock."

# Some of the key services and tools that Amazon provides for developers include:

# 1. Amazon Web Services (AWS): This is Amazon's comprehensive cloud computing platform, offering a wide range of services such as computing power, storage, databases, networking, machine learning, analytics, and more.

# 2. AWS Lambda: A serverless computing service that allows developers to run code without provisioning or managing servers.

# 3. AWS CodeCommit, CodeBuild, CodeDeploy, and CodePipeline: These services help developers with source control, building, testing, and deploying applications on AWS.

# 4. AWS CodeStar: An integrated service that helps developers quickly develop, build, and deploy applications on AWS.  

# 5. AWS Amplify: A set of tools and services that help developers build and deploy secure, scalable, and full-stack web and mobile applications.

# 6. AWS SDK: Software Development Kits (SDKs) that allow developers to build applications using various programming languages and integrate them with AWS services.

# 7. AWS CodeWhisperer: An AI-powered code review and code generation service that can help developers write and improve code faster.

# To use these services and tools, developers typically sign up for an AWS account, select the services they need, and integrate them into their development workflows and applications using the provided SDKs, APIs, and documentation.     

#  and mobile applications.

# 6. AWS SDK: Software Development Kits (SDKs) that allow developers to build applications using various programming languages and integrate them with AWS services.

# 7. AWS CodeWhisperer: An AI-powered code review and code generation service that can help developers write and improve code faster.

# To use these services and tools, developers typically sign up for an AWS account, select the services they need, and integrate them into their development workflows and applications using the provided SDKs, APIs, and documentation.     


# 7. AWS CodeWhisperer: An AI-powered code review and code generation service that can help developers write and improve code faster.

# To use these services and tools, developers typically sign up for an AWS account, select the services they need, and integrate them into their development workflows and applications using the provided SDKs, APIs, and documentation.     

# ntegrate them into their development workflows and applications using the provided SDKs, APIs, and documentation.     

# If you're looking for specific information on how to use a particular AWS service or tool, it would be best to consult the official AWS documentation or seek guidance from experienced AWS developers.
# Latency: 9.4386 seconds

# --------------------------------------------------------------------------------
# Prompt: What is the pricing model for Amazon Bedrock?

# Model: amazon.nova-lite-v1:0
# Response: Amazon Bedrock is a service provided by Amazon Web Services (AWS) that allows developers to build, test, and scale generative AI applications. As of the latest updates, the pricing model for Amazon Bedrock is typically based on the following components:

# 1. **Model Inference Costs**:
#    - **Per-Request Pricing**: Charges are incurred based on the number of inference requests made to the models. The pricing varies depending on the specific model and the type of request (e.g., text generation, embeddings).
#    - **Model Tiers**: Different models may have different pricing tiers. For instance, more advanced or high-performance models may cost more per request compared to standard models.

# 2. **Data Processing Costs**:
#    - **Input Data Processing**: Costs associated with processing input data before it is sent to the model. This could include data transformation, preprocessing, and tokenization.
#    - **Output Data Processing**: Costs for processing the model's output after inference, such as post-processing, formatting, and any additional steps required to prepare the final response.

# 3. **Storage Costs**:
#    - **Model Storage**: Costs for storing the models and any associated artifacts, such as checkpoints and training data.
#    - **Output Storage**: Costs for storing the generated outputs, which may be necessary for logging, auditing, or further analysis.

# 4. **Additional Costs**:
#    - **Custom Model Costs**: If you develop and deploy custom models, there may be additional costs associated with the resources used for training and fine-tuning those models.
#    - **Data Transfer Costs**: Costs for data transfer in and out of the service, especially if you are working with large datasets or generating a high volume of responses.

# ### Example Pricing (Hypothetical as of 2023)

# - **Text Generation Model (Standard Tier)**:
#   - **Input Request**: $0.002 per 1,000 characters
#   - **Output Request**: $0.003 per 1,000 characters

# - **Embedding Model (Advanced Tier)**:
#   - **Input Request**: $0.005 per 1,000 tokens
#   - **Output Request**: $0.007 per 1,000 tokens

# ### Usage Examples

# - **Generating 1,000 characters of text**:
#   - Standard Tier: \((1,000
# Latency: 4.2811 seconds

# Model: anthropic.claude-3-sonnet-20240229-v1:0
# Response: Unfortunately, I do not have any specific information about pricing for Amazon Bedrock. Amazon Bedrock appears to be an internal codename or project name used by Amazon, but it is not a publicly released product or service that I could find pricing details for.

# Amazon has many different cloud services like AWS, Amazon Web Services, with published pricing models. But without more context about what exactly Amazon Bedrock refers to, I cannot provide accurate pricing information. Amazon likely keeps details about internal projects and codenames confidential unless officially announced as a commercial product or 
# service.
# Latency: 4.2945 seconds

# --------------------------------------------------------------------------------