!pip install transformers torch sentence-transformers
!pip install -U datasets

from transformers import pipeline
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import torch

# Step 1: Load the TruthfulQA dataset
# We'll use the 'generation' subset for simplicity, which has 'question' and 'best_answer' fields
dataset = load_dataset("truthful_qa", "generation")

# Step 2: Initialize the Sentence Transformer model
# This model will be used to embed both the questions and the potential answers
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 3: Index the answers
# We will embed the best answers from the training set to create a small knowledge base
# For a real RAG system, this would be a much larger index of documents
train_answers = dataset['validation']['best_answer']
train_embeddings = model.encode(train_answers, convert_to_tensor=True)

# Step 4: Set up a retrieval function
def retrieve_answer(query, embeddings, answers, top_k=1):
  """
  Retrieves the most similar answer from the knowledge base to the query.
  """
  query_embedding = model.encode(query, convert_to_tensor=True)
  # Calculate cosine similarity between the query and all answer embeddings
  cosine_scores = util.cos_sim(query_embedding, embeddings)[0]
  # Get the top k most similar answers
  top_results = torch.topk(cosine_scores, k=top_k)

  retrieved_answers = [answers[idx] for idx in top_results.indices]
  return retrieved_answers

# Import Sentence-Transformers utility for cosine similarity
from sentence_transformers import util

# Step 5: Set up a simple generation model (optional, but part of RAG)
# For this example, we'll use a basic text generation pipeline
# In a full RAG system, this generator would take the query AND retrieved context
generator = pipeline("text-generation", model="gpt2") # Using a smaller model for faster inference

# Step 6: Create the RAG process
def simple_rag(query, embeddings, answers, generator_pipeline, top_k_retrieval=1):
  """
  Performs a simple Retrieval Augmented Generation process.
  """
  # Retrieve relevant context
  retrieved_context = retrieve_answer(query, embeddings, answers, top_k=top_k_retrieval)

  # Combine query and context for generation
  # A simple way is to prepend the retrieved context to the query
  # A more sophisticated approach would structure the prompt carefully
  prompt = f"Context: {'. '.join(retrieved_context)}\n\nQuestion: {query}\nAnswer:"

  # Generate the answer based on the prompt
  # We set max_new_tokens to avoid overly long responses and do_sample=True for variability
  generated_output = generator_pipeline(prompt, max_new_tokens=50, do_sample=True)[0]['generated_text']

  # In a more robust RAG, you might post-process the generated output
  return generated_output

# Step 7: Test the RAG system with a question from the dataset
# We'll pick a question from the validation set that wasn't used for indexing
test_question = dataset['validation'][0]['question']
print(f"Query: {test_question}")

# Run the simple RAG process
generated_answer = simple_rag(test_question, train_embeddings, train_answers, generator, top_k_retrieval=1)
print(f"Generated Answer: {generated_answer}")

# You can try with other questions
# test_question_2 = "What is the capital of France?" # Example of a general knowledge question
# generated_answer_2 = simple_rag(test_question_2, train_embeddings, train_answers, generator, top_k_retrieval=1)
# print(f"Query: {test_question_2}")
# print(f"Generated Answer: {generated_answer_2}")
# replace gpt-2 with Google's Gemma Model
