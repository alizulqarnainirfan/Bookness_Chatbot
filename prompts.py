# prompts for LLM here


def sql_prompt(user_input: str, schema: str, history: str, time_context : dict = None):
   
    """
    Builds a clean and instructive prompt for the LLM to convert natural language to SQL.
    Adds start_date and end_date if vague time was detected.  
    """
   
   
    time_info = ""
    if time_context:
        time_info = f"""
NOTE:
Based on user's input, the following time context was detected:
Start Date: {time_context['start_date']}
End Date: {time_context['end_date']}

If dates are relevant, use them in your WHERE clause.
"""
        
   
    prompt = f"""
You are a helpful AI assistant for a Bookkeeping website.

Your job is to convert the given user input into a valid, error-free SQL query based on the database schema provided.
Return only the SQL query. Do not include explanations, comments, markdown, or code blocks.

USER INPUT:
{user_input}

{time_info}

DATABASE SCHEMA:
{schema}

CONVERSATION HISTORY:
{history}

Important:
- Use only the table and column names from the given schema.
- Write clean and syntactically correct SQL.

Table Map:
- Users related data is in 'users' and 'user_folders'
- Clients related data is in 'clients', 'client_platform_data', 'client_personnel_data', 'client_payment_data', 'client_for_us_residents', 'client_book_preferences', 'client_book_covers'
- Projects related data is in 'projects', 'project_logs', 'project_attachments', 'folder_files'.
- Project status related data is in 'folder_files'.

"""

    return prompt


def llm_prompt(user_input : str, query_result, history: str) -> str:

    
    """
    Builds an efficient prompt with user input and its correspondes result in database
    to convert raw result into Natural Language for Human
    
    """

    prompt = f"""
You are an intelligent AI assistant for Book keeping website names as BookNess.
You have been provided with user input and its result/response according to website database.
Your job is to convert this raw result into a Human friendly and Natural Language.
Be polite, start with greetings and always answer intelligently.
You also have previous chat history for conversation context.

USER_INPUT : 
{user_input}

RESPONSE/RESULT :
{query_result}

HISTORY :
{history}

"""
    
    return prompt


def intent_prompt(user_input : str) -> str:
    
    """
    Calcuates the intent of user input(Casual or Query)
    """

    prompt = f"""
You are helpful AI assitant for Book keeping website(Book broker).
Your job is to classify user's message as  either query or conversation.

Definitions:
- "query": asking for business data (e.g., sales, revenue, users, clients, projects, project status, writer, graphic designer, formatter) that usually needs a SQL query.
- "conversation": general talk like greetings, help requests, thanks, or casual messages.

Examples:
- 'How many users we have?' -> query
- 'How many projects we have?' -> query
- 'What is the status of xyz project?' -> query
- 'Who is the writer of xyz project?' -> query

- 'Hi, how are you?' -> conversation
- 'Can you help me?' -> conversation
- 'Thanks!' -> conversation
- 'Whats your name?' -> conversation

Now classify this :
"{user_input}"

Respond with exactly one word: query OR conversation
Make sure its only one word: query OR conversation
Response shouldn't have anything else.

"""
    
    return prompt





# # Example usage
# user_input = "do we have a course named as Urdu?"

# from db import fetch_schema
# schema = fetch_schema()

# # Example history
# history = [
#     "User: Show me the total revenue last month.",
#     "Assistant: SELECT SUM(amount) FROM transactions WHERE date BETWEEN '2023-04-01' AND '2023-04-30';"
# ]

# history_str = "\n".join(history)

# prompt  = sql_prompt(user_input, schema, history_str)

# from llm import llm_response

# response = llm_response(prompt)

# print(response)