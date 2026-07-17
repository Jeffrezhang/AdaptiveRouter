import psycopg2
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

examples = [
    # ── code_generation (large) ───────────────────────────────────────────────
    # Core implementation requests
    ("write a python function to sort a list", "code_generation"),
    ("implement binary search in javascript", "code_generation"),
    ("build a REST API endpoint in fastapi", "code_generation"),
    ("create a class for managing users", "code_generation"),
    ("generate a sql query to find duplicates", "code_generation"),
    ("write a bash script to backup files", "code_generation"),
    ("implement a linked list in java", "code_generation"),
    ("create a react component for a navbar", "code_generation"),
    ("write unit tests for this function", "code_generation"),
    ("build a websocket server in python", "code_generation"),
    # Sorting algorithms
    ("implement bubble sort in python", "code_generation"),
    ("implement merge sort in javascript", "code_generation"),
    ("implement quicksort in javascript", "code_generation"),
    ("implement insertion sort in java", "code_generation"),
    ("implement heapsort in python", "code_generation"),
    ("write a sorting algorithm from scratch", "code_generation"),
    ("create a function that sorts a list of numbers", "code_generation"),
    ("implement a list sorting algorithm in python", "code_generation"),
    ("build a function that takes a list and returns it sorted", "code_generation"),
    ("generate python code to sort a list of numbers", "code_generation"),
    # Data structures
    ("implement a linked list in javascript", "code_generation"),
    ("implement a stack data structure in javascript", "code_generation"),
    ("implement a queue in javascript", "code_generation"),
    ("implement a hash map in javascript", "code_generation"),
    ("implement a binary tree in python", "code_generation"),
    ("implement a priority queue in python", "code_generation"),
    ("implement a graph data structure in java", "code_generation"),
    ("implement a trie data structure in python", "code_generation"),
    # Search algorithms
    ("implement binary search in python", "code_generation"),
    ("implement binary search in java", "code_generation"),
    ("implement binary search in typescript", "code_generation"),
    ("implement depth first search in javascript", "code_generation"),
    ("implement breadth first search in javascript", "code_generation"),
    ("implement dijkstra algorithm in python", "code_generation"),
    ("implement a pathfinding algorithm", "code_generation"),
    # Web / API
    ("build a REST API with authentication", "code_generation"),
    ("create a react component for a login form", "code_generation"),
    ("write a graphql resolver in nodejs", "code_generation"),
    ("build a middleware function for express", "code_generation"),
    ("create a database migration script", "code_generation"),
    ("write a cron job to clean up old files", "code_generation"),
    ("generate a dockerfile for a python app", "code_generation"),
    ("write a github actions workflow for ci", "code_generation"),
    ("create a websocket chat server", "code_generation"),
    ("build a rate limiter in python", "code_generation"),
    # ML / data
    ("write a script to train a linear regression model", "code_generation"),
    ("implement k-means clustering from scratch", "code_generation"),
    ("build a data preprocessing pipeline", "code_generation"),
    ("create a csv parser in python", "code_generation"),
    ("write a script to scrape a website", "code_generation"),
    ("generate code to visualize data with matplotlib", "code_generation"),
    ("implement a neural network layer from scratch", "code_generation"),
    ("write a tokenizer for a text dataset", "code_generation"),
    ("build an embedding search system", "code_generation"),
    ("create a recommendation algorithm", "code_generation"),
    # General coding tasks
    ("write a recursive function to compute fibonacci", "code_generation"),
    ("create a decorator in python that logs function calls", "code_generation"),
    ("implement a pub-sub system in python", "code_generation"),
    ("write a thread-safe singleton in java", "code_generation"),
    ("implement the observer pattern in typescript", "code_generation"),
    ("create a command line tool with argparse", "code_generation"),
    ("write a script that monitors file changes", "code_generation"),
    ("build a simple compiler for arithmetic expressions", "code_generation"),
    ("implement a lru cache in python", "code_generation"),
    ("write code to parse and validate json", "code_generation"),

    # ── code_understanding (small) ────────────────────────────────────────────
    ("explain what this recursive function does", "code_understanding"),
    ("why is my code throwing a null pointer exception", "code_understanding"),
    ("what does this algorithm do", "code_understanding"),
    ("review my pull request and give feedback", "code_understanding"),
    ("walk me through this code snippet", "code_understanding"),
    ("what is the time complexity of this function", "code_understanding"),
    ("why does this loop run forever", "code_understanding"),
    ("what does the yield keyword do in python", "code_understanding"),
    ("explain this regex pattern", "code_understanding"),
    ("debug this stack trace for me", "code_understanding"),
    ("explain what this existing function does", "code_understanding"),
    ("what does this code snippet do when I run it", "code_understanding"),
    ("read through this function and tell me what it does", "code_understanding"),
    ("I have this code already written, explain it to me", "code_understanding"),
    ("look at this existing code and describe its behavior", "code_understanding"),
    ("what does this line of code mean", "code_understanding"),
    ("why is this function returning the wrong value", "code_understanding"),
    ("what is wrong with this code", "code_understanding"),
    ("can you spot the bug in this function", "code_understanding"),
    ("explain the difference between these two implementations", "code_understanding"),
    ("what does async await do in javascript", "code_understanding"),
    ("why is my api returning a 500 error", "code_understanding"),
    ("what is the purpose of this class", "code_understanding"),
    ("explain what a closure is in javascript", "code_understanding"),
    ("why is my variable undefined", "code_understanding"),
    ("what does this error message mean", "code_understanding"),
    ("how does this sorting algorithm work", "code_understanding"),
    ("explain what this sql query returns", "code_understanding"),
    ("what is the difference between == and === in javascript", "code_understanding"),
    ("why is my recursive function hitting max recursion depth", "code_understanding"),

    # ── technical_design (large) ──────────────────────────────────────────────
    ("how should I design a microservices architecture", "technical_design"),
    ("what are the tradeoffs between postgres and mongodb", "technical_design"),
    ("should I use kafka or rabbitmq", "technical_design"),
    ("design a database schema for ecommerce", "technical_design"),
    ("how do I structure a large react application", "technical_design"),
    ("what design pattern should I use here", "technical_design"),
    ("how should I handle authentication in my app", "technical_design"),
    ("compare REST vs GraphQL for my use case", "technical_design"),
    ("how do I scale this system to millions of users", "technical_design"),
    ("what caching strategy should I use", "technical_design"),
    ("design a system for real-time notifications", "technical_design"),
    ("how should I architect a multi-tenant saas app", "technical_design"),
    ("what is the best way to handle file uploads at scale", "technical_design"),
    ("how do I design an api that supports versioning", "technical_design"),
    ("what database should I choose for time series data", "technical_design"),
    ("how should I structure my ci cd pipeline", "technical_design"),
    ("design a search system for millions of documents", "technical_design"),
    ("how do I handle distributed transactions", "technical_design"),
    ("what load balancing strategy should I use", "technical_design"),
    ("how do I design a fault tolerant system", "technical_design"),
    ("compare sql vs nosql for my use case", "technical_design"),
    ("how should I implement role based access control", "technical_design"),
    ("design an event driven architecture", "technical_design"),
    ("how do I prevent race conditions in my api", "technical_design"),
    ("what monitoring strategy should I use for my service", "technical_design"),

    # ── analytical_reasoning (large) ──────────────────────────────────────────
    ("calculate the probability of flipping heads 5 times", "analytical_reasoning"),
    ("solve this differential equation", "analytical_reasoning"),
    ("if I have 100 users and 10% churn what is retention after 6 months", "analytical_reasoning"),
    ("prove that the square root of 2 is irrational", "analytical_reasoning"),
    ("what is the expected value of this random variable", "analytical_reasoning"),
    ("optimize this algorithm for minimum time complexity", "analytical_reasoning"),
    ("given these constraints find the optimal solution", "analytical_reasoning"),
    ("analyze the time and space complexity", "analytical_reasoning"),
    ("calculate compound interest over 10 years", "analytical_reasoning"),
    ("is 9.9 bigger than 9.11", "analytical_reasoning"),
    ("which number is larger 0.5 or 0.50", "analytical_reasoning"),
    ("compare these two decimal numbers", "analytical_reasoning"),
    ("which is greater 1.9 or 1.10", "analytical_reasoning"),
    ("what is the probability of this event occurring", "analytical_reasoning"),
    ("what is the derivative of x squared", "analytical_reasoning"),
    ("solve this system of linear equations", "analytical_reasoning"),
    ("what is the integral of sin x", "analytical_reasoning"),
    ("prove this mathematical theorem", "analytical_reasoning"),
    ("what is the big o notation of this algorithm", "analytical_reasoning"),
    ("calculate the variance of this dataset", "analytical_reasoning"),
    ("what is the bayesian probability of this outcome", "analytical_reasoning"),
    ("analyze this logic puzzle and find the solution", "analytical_reasoning"),
    ("if train a leaves at 9am and train b at 10am when do they meet", "analytical_reasoning"),
    ("what is the optimal strategy for this game theory problem", "analytical_reasoning"),
    ("calculate the roi of this investment over 5 years", "analytical_reasoning"),
    ("what is the p value and what does it mean", "analytical_reasoning"),
    ("explain the monty hall problem", "analytical_reasoning"),
    ("what is the birthday paradox probability", "analytical_reasoning"),
    ("solve this combinatorics problem", "analytical_reasoning"),
    ("analyze the tradeoffs between these two approaches quantitatively", "analytical_reasoning"),
    ("what is the computational complexity of this recursive algorithm", "analytical_reasoning"),
    ("calculate the confidence interval for this sample", "analytical_reasoning"),
    ("prove by induction that this formula holds", "analytical_reasoning"),
    ("find the minimum spanning tree of this graph", "analytical_reasoning"),
    ("what is the time complexity of mergesort", "analytical_reasoning"),

    # ── writing (small) ───────────────────────────────────────────────────────
    ("draft a professional email to my team about a project delay", "writing"),
    ("write a blog post about machine learning", "writing"),
    ("compose a cover letter for a software engineer role", "writing"),
    ("rewrite this paragraph to be more concise", "writing"),
    ("help me write a persuasive essay", "writing"),
    ("proofread and improve this article", "writing"),
    ("write a product description for my app", "writing"),
    ("draft a performance review for my employee", "writing"),
    ("write a LinkedIn post about my new project", "writing"),
    ("summarize this document in 3 bullet points", "writing"),
    ("write a thank you email to my interviewer", "writing"),
    ("help me write a resignation letter", "writing"),
    ("draft a project proposal for my manager", "writing"),
    ("write a tweet about my new feature launch", "writing"),
    ("improve the tone of this email to sound more professional", "writing"),
    ("write a short bio for my website", "writing"),
    ("draft meeting notes from this agenda", "writing"),
    ("write a press release for our product launch", "writing"),
    ("help me write a cold outreach email", "writing"),
    ("rewrite this job description to be more inclusive", "writing"),

    # ── factual_lookup (small) ────────────────────────────────────────────────
    ("what is the capital of France", "factual_lookup"),
    ("who invented the telephone", "factual_lookup"),
    ("when did world war 2 end", "factual_lookup"),
    ("what is the population of Tokyo", "factual_lookup"),
    ("define photosynthesis", "factual_lookup"),
    ("how tall is mount everest", "factual_lookup"),
    ("what year was python created", "factual_lookup"),
    ("who is the CEO of Apple", "factual_lookup"),
    ("what is the speed of light", "factual_lookup"),
    ("when was the Eiffel Tower built", "factual_lookup"),
    ("what is the boiling point of water", "factual_lookup"),
    ("who wrote hamlet", "factual_lookup"),
    ("what is the largest country by area", "factual_lookup"),
    ("how many planets are in the solar system", "factual_lookup"),
    ("what does DNA stand for", "factual_lookup"),
    ("who painted the mona lisa", "factual_lookup"),
    ("what is the currency of japan", "factual_lookup"),
    ("how long is the great wall of china", "factual_lookup"),
    ("what year did the berlin wall fall", "factual_lookup"),
    ("what is the atomic number of carbon", "factual_lookup"),

    # ── general (small) ───────────────────────────────────────────────────────
    ("tell me a joke", "general"),
    ("recommend a good movie to watch", "general"),
    ("what should I have for dinner", "general"),
    ("can you help me with something", "general"),
    ("what do you think about this idea", "general"),
    ("I need some advice", "general"),
    ("what are some fun weekend activities", "general"),
    ("help me decide between two options", "general"),
    ("what is your opinion on this", "general"),
    ("give me some inspiration", "general"),
    ("who are you", "general"),
    ("what are some good books to read", "general"),
    ("how do I stay motivated", "general"),
    ("what are some tips for better sleep", "general"),
    ("recommend a podcast about technology", "general"),
    ("what should I know before traveling to japan", "general"),
    ("how do I make friends as an adult", "general"),
    ("what are some healthy breakfast ideas", "general"),
    ("give me a fun fact", "general"),
    ("what are some hobbies I could pick up", "general"),
]

conn = psycopg2.connect(
    host="127.0.0.1", port=5432,
    dbname="routing", user="litellm", password="litellm"
)
cur = conn.cursor()

cur.execute("DELETE FROM request_examples")

print(f"Embedding {len(examples)} examples...")
texts = [ex[0] for ex in examples]
embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32)

for (text, request_type), embedding in zip(examples, embeddings):
    cur.execute(
        "INSERT INTO request_examples (text, request_type, embedding) VALUES (%s, %s, %s)",
        (text, request_type, embedding.tolist())
    )

conn.commit()
cur.close()
conn.close()

from collections import Counter
counts = Counter(ex[1] for ex in examples)
print(f"\nInserted {len(examples)} examples:")
for cat, count in sorted(counts.items()):
    print(f"  {cat:<25} {count}")
print("\nDone.")