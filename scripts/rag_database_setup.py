from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()

# Access the Pinecone API key from environment variables
api_key = os.getenv("PINECONE_API_KEY")

# Check if the API key is available
if api_key is None:
    raise ValueError("PINECONE_API_KEY not found. Please check your .env file")

# Initialize Pinecone client with the API key
pc = Pinecone(api_key=api_key)

# Define the index name and configuration
index_name = "basic-rag-project"
dimension = 384  # Adjust this to match your embedding dimension
metric = "cosine"  # Define the metric to be used (e.g., cosine similarity)

# Define the spec for the serverless index with cloud and region
spec = ServerlessSpec(
    cloud="aws",  # Cloud provider
    region="us-east-1"  # Use a valid region supported by your plan
)

# Check if the index already exists
if index_name not in pc.list_indexes().names():
    # Create the index
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=spec  # Provide the region and cloud provider configuration
    )
    print(f"Index '{index_name}' created successfully.")
else:
    print(f"Index '{index_name}' already exists.")
