"""Prompt dataset generation for different task types."""

from typing import List, Dict

# Sample prompts for different task types
PROMPT_TEMPLATES = {
    'code': [
        "Write a Python function to calculate the fibonacci sequence up to n terms.",
        "Implement a binary search algorithm in JavaScript.",
        "Create a class in Java that represents a linked list with insert and delete methods.",
        "Write a SQL query to find the top 10 customers by total purchase amount.",
        "Debug this code: for i in range(10) print(i)",
        "Implement a function to reverse a string without using built-in methods.",
        "Write a Python decorator that measures function execution time.",
        "Create a React component that fetches and displays user data from an API.",
        "Implement quicksort in C++.",
        "Write a function to detect cycles in a directed graph.",
        "Create a REST API endpoint in Node.js that handles user authentication.",
        "Write a regex pattern to validate email addresses.",
        "Implement a LRU cache in Python.",
        "Create a SQL migration to add a foreign key constraint.",
        "Write a function to find the longest common subsequence of two strings.",
        "Implement a thread-safe singleton pattern in Java.",
        "Create a function that merges two sorted arrays.",
        "Write a parser for JSON using recursive descent.",
        "Implement depth-first search for a binary tree.",
        "Create a function to check if a string is a valid palindrome.",
    ],
    'math': [
        "What is the derivative of x^3 + 2x^2 - 5x + 7?",
        "Solve the equation: 2x + 5 = 3x - 7",
        "Calculate the area of a circle with radius 5cm.",
        "What is the integral of sin(x) from 0 to π?",
        "Find the eigenvalues of the matrix [[2, 1], [1, 2]].",
        "If a train travels 120 miles in 2 hours, what is its average speed?",
        "Solve the system of equations: x + y = 5 and 2x - y = 1",
        "What is the probability of rolling two dice and getting a sum of 7?",
        "Calculate the volume of a sphere with radius 3.",
        "Find the roots of x^2 - 5x + 6 = 0",
        "What is the Taylor series expansion of e^x around x=0?",
        "Calculate the determinant of [[1, 2, 3], [4, 5, 6], [7, 8, 9]].",
        "What is the limit of (x^2 - 1)/(x - 1) as x approaches 1?",
        "Find the inverse of the matrix [[1, 2], [3, 4]].",
        "Calculate the sum of the first 100 natural numbers.",
        "What is the Fourier transform of a constant function?",
        "Solve the differential equation dy/dx = 2x.",
        "Calculate the permutations of choosing 3 items from 10.",
        "What is the geometric mean of 4, 16, and 64?",
        "Find the angle between vectors [1, 2, 3] and [4, 5, 6].",
    ],
    'creative': [
        "Write a short story about a robot learning to paint.",
        "Describe a sunset on an alien planet.",
        "Create a poem about the changing seasons.",
        "Write a dialogue between two old friends meeting after 20 years.",
        "Imagine a world where everyone can read minds. Describe daily life.",
        "Write the opening paragraph of a mystery novel.",
        "Describe your ideal vacation destination in vivid detail.",
        "Create a character profile for a detective with unusual abilities.",
        "Write a letter from the future to your present self.",
        "Describe the taste of your favorite childhood food.",
        "Write a haiku about technology.",
        "Create a fantasy creature and describe its habitat.",
        "Write a dramatic monologue for a villain explaining their motives.",
        "Describe a bustling marketplace in a medieval city.",
        "Create a love story that begins with a mistake.",
        "Write about a musician who has lost their ability to hear.",
        "Describe the feeling of discovering something new.",
        "Create a dystopian society where books are forbidden.",
        "Write a conversation between the moon and the sun.",
        "Describe the last day on Earth from multiple perspectives.",
    ],
    'factual': [
        "What is the capital of France?",
        "Who was the first president of the United States?",
        "What year did World War II end?",
        "What is the largest planet in our solar system?",
        "Who wrote 'Romeo and Juliet'?",
        "What is the speed of light?",
        "How many elements are in the periodic table?",
        "What is the deepest ocean on Earth?",
        "Who invented the telephone?",
        "What is the longest river in the world?",
        "When was the Declaration of Independence signed?",
        "What is the boiling point of water at sea level?",
        "Who painted the Mona Lisa?",
        "What is the smallest country in the world?",
        "How many chromosomes do humans have?",
        "What is the tallest mountain on Earth?",
        "Who discovered penicillin?",
        "What year did the Berlin Wall fall?",
        "What is the currency of Japan?",
        "How many bones are in the human body?",
    ],
    'reasoning': [
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "Three friends split a bill. Alice paid $20, Bob paid $15, and Carol paid $25. If they want to split it evenly, who owes whom?",
        "A farmer has 17 sheep. All but 9 die. How many are left?",
        "If you're running a race and you pass the person in 2nd place, what place are you in?",
        "A doctor gives you three pills and tells you to take one every half hour. How long will they last?",
        "If there are 3 apples and you take away 2, how many do you have?",
        "Can a man legally marry his widow's sister?",
        "You have a 3-gallon jug and a 5-gallon jug. How do you measure exactly 4 gallons?",
        "If a plane crashes on the border of US and Canada, where do you bury the survivors?",
        "A clerk in a butcher shop is 5'10\" tall. What does he weigh?",
        "How many times can you subtract 10 from 100?",
        "Some months have 31 days, some have 30. How many have 28?",
        "If a red house is made of red bricks and a blue house is made of blue bricks, what is a greenhouse made of?",
        "Two coins add up to 30 cents. One is not a nickel. What are they?",
        "You enter a dark room with a match. There's a candle, oil lamp, and fireplace. What do you light first?",
        "How can a man go 8 days without sleep?",
        "What occurs once in a minute, twice in a moment, but never in a thousand years?",
        "If you have a bowl with 6 apples and you take away 4, how many do you have?",
    ]
}


def get_prompts_by_type(prompt_type: str, num_samples: int) -> List[str]:
    """Get sample prompts for a specific type.

    Args:
        prompt_type: Type of prompts ('code', 'math', 'creative', 'factual', 'reasoning')
        num_samples: Number of samples to return

    Returns:
        List of prompt strings
    """
    if prompt_type not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt type: {prompt_type}. "
                        f"Available types: {list(PROMPT_TEMPLATES.keys())}")

    templates = PROMPT_TEMPLATES[prompt_type]

    # If requesting more samples than templates, cycle through them
    if num_samples <= len(templates):
        return templates[:num_samples]
    else:
        # Repeat the templates
        full_cycles = num_samples // len(templates)
        remainder = num_samples % len(templates)
        return templates * full_cycles + templates[:remainder]


def get_all_prompt_types() -> List[str]:
    """Get list of all available prompt types."""
    return list(PROMPT_TEMPLATES.keys())


def load_prompts_dataset(config) -> Dict[str, List[str]]:
    """Load prompts for all configured types.

    Args:
        config: Configuration object with data.prompt_types and data.samples_per_type

    Returns:
        Dictionary mapping prompt_type -> list of prompts
    """
    dataset = {}

    for prompt_type in config.data.prompt_types:
        prompts = get_prompts_by_type(prompt_type, config.data.samples_per_type)
        dataset[prompt_type] = prompts
        print(f"Loaded {len(prompts)} prompts for type '{prompt_type}'")

    return dataset
