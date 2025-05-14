# Main logic of API and code here

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from db import fetch_schema, execute_query
from prompts import sql_prompt, llm_prompt, intent_prompt
from utils import cleaned_sql, is_safe_sql, parse_vague_time_phrases
from llm import llm_response, llm_response_stream
from memory import get_history, append_to_history, clear_history


app = FastAPI(title="Bookness Chatbot", description="Smart AI assistant for Bookness", version="1.0")


app.add_middleware(
    
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]

)


class ChatRequest(BaseModel):

    session_id : str
    user_input : str


# Root End point
@app.get("/")

def root():

    return {
        "Message": "Booknes Chatbot is Running like FLASH ⚡"
    }


# Chat End Point
@app.post("/chat")
async def chat_endpoint(payload : ChatRequest):

    try:
        # Get session ID and User_input from input -> post (chat endpoint)
        session_id = payload.session_id
        user_input = payload.user_input

        if not user_input:
            return JSONResponse(status_code = 480, content = {"Error": " Missing 'query' field in the request"})



        # Getting the paramaters which we gonna pass to the Necessary functions
        schema = fetch_schema()
        history_list = get_history(session_id)
        history_str = "\n".join(history_list)


        # Lets calculate the intent of User's Message
        intent = llm_response(intent_prompt(user_input)).lower().strip()

        # Chat handling after calculating the intent of User's Message
        if intent == "query":

            # Detect vague time expressions like "last month", "this year"
            time_context = parse_vague_time_phrases(user_input)

            sql = llm_response(sql_prompt(user_input, schema, history_str, time_context)).strip()
            sql = cleaned_sql(sql)

            if not sql:
                return {"Response": "I couldn't generate a valid SQL query. Please repharase." }
            
            if not is_safe_sql(sql):
                return {"Response": "Unsafe command detected ❌ Sorry, I'm not allowed to perform these type of tasks"}
            
            result = execute_query(sql)
            prompt = llm_prompt(user_input=user_input, query_result=result, history=history_str)

            # Streaming Gemini response
            def stream_gen():

                full_response = ""
                for chunk in llm_response_stream(prompt):
                    full_response += chunk
                    yield chunk

                append_to_history(session_id, user_input, full_response)

            return StreamingResponse(stream_gen(), media_type="text/plain")

        elif intent == "conversation":

            def stream_gen():
            
                full_response = ""
                for chunk in llm_response_stream(user_input):
                    full_response += chunk
                    yield chunk
            
                append_to_history(session_id, user_input, full_response)

            return StreamingResponse(stream_gen(), media_type="text/plain")

        else:
            return {"Response": "Sorry I couldn't determine your intent. Try rephrasing"}

    except Exception as e:
        return {"Error": str(e)}
    













    







































# session_histories = {}

# user_input = ""

# schema = fetch_schema()

# while user_input.lower() != 'exit':

#     user_input = input("Hello from Bookness Chatbot. How can I help you (write exit to quit conversation): ")

#     prompt = intent_prompt(user_input).lower()
#     intent = llm_response(prompt).strip()

#     print(intent)

#     if intent == 'query':
      
#         sql = llm_response(sql_prompt(user_input, schema, history=""))
      
#         sql = cleaned_sql(sql)
        
#         if not sql:
#             print("Unable to get the sql")
        
#         if not is_safe_sql(sql):
#             print("Access Denied! Can't perform this kind of operation")
        
#         result = execute_query(sql)
       
#         natural_response = llm_response(llm_prompt(user_input=user_input, query_result= result, history= ""))
       
#         print(natural_response)
    

#     elif intent == 'conversation':
      
#         response = llm_response(user_input).strip()
#         print(response)

#     else:
      
#         print("Sorry, failed in calculating Intent. Can you rephrase your message")

        












# print("Starting script...")

# import os
# import io
# # from PIL import Image as PILImppage
# from typing import Annotated, TypedDict, Any, List, Dict
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import AnyMessage, add_messages
# from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
# # from IPython.display import Image, display
# from langchain_core.runnables.graph import MermaidDrawMethod
# from langchain.tools import tool
# from dotenv import load_dotenv
# import mysql.connector
# from mysql.connector import pooling
# from mysql.connector import Error, errorcode
# import sys
# import time
# import google.generativeai as genai
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph import MessagesState
# from tenacity import retry, stop_after_attempt, wait_exponential

# # Load environment variables
# load_dotenv(override=True)
# print("Environment variables loaded")

# # Initialize Gemini client
# gemini_key = os.getenv('GEMINI_API_KEY')
# if not gemini_key:
#     raise ValueError("GEMINI_API_KEY environment variable not set")
# genai.configure(api_key=gemini_key)
# model = genai.GenerativeModel('gemini-1.5-flash')

# # Remove the connection_pool initialization from the global scope
# connection_pool = None

# def get_connection():
#     """Get a connection from the pool (lazy initialization)"""
#     global connection_pool
#     try:
#         if connection_pool is None:
#             connection_pool = mysql.connector.pooling.MySQLConnectionPool(
#                 pool_name="mitrat_pool",
#                 pool_size=5,
#                 host=os.getenv('DB_HOST'),
#                 user=os.getenv('DB_USER'),
#                 password=os.getenv('DB_PASSWORD'),
#                 database=os.getenv('DB_NAME'),
#                 port=3306,
#                 connect_timeout=30,
#                 raise_on_warnings=True,
#                 # use_pure=True,
#                 auth_plugin='mysql_native_password',
#                 # ssl_disabled=True
#             )
#         return connection_pool.get_connection()
#     except mysql.connector.Error as e:
#         if e.errno == errorcode.CR_CONN_HOST_ERROR:
#             raise Exception("Server IP not whitelisted. Please add this server's IP to the database whitelist.")
#         else:
#             raise Exception(f"Database connection failed: {str(e)}")

# # Cache schemas after the first retrieval
# schema_cache = {}

# @tool
# def get_schema_tool(table_name: str) -> str:
#     """Get the schema and sample data for a specific table"""
#     if table_name in schema_cache:
#         return schema_cache[table_name]
    
#     print(f"[Tool Log] get_schema_tool called with table_name={table_name}")
    
#     try:
#         connection = get_connection()
#         cursor = connection.cursor(dictionary=True)
        
#         # Get schema
#         cursor.execute(f"SHOW CREATE TABLE {table_name}")
#         schema_result = cursor.fetchone()
#         schema = schema_result['Create Table']
        
#         # Get sample data (first 5 rows)
#         cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
#         sample_data = cursor.fetchall()
        
#         # Format the response
#         response = f"Table: {table_name}\n\nSCHEMA:\n{schema}\n\nSAMPLE DATA:"
#         for row in sample_data:
#             response += f"\n{row}"
            
#         cursor.close()
#         connection.close()
        
#         # Cache the schema
#         schema_cache[table_name] = response
#         return response
        
#     except Exception as e:
#         return f"Error retrieving schema and data: {str(e)}"

# @tool
# def execute_query_tool(query: str) -> List[Dict[str, Any]]:
#     """Execute a SQL query using a connection from the pool"""
#     connection = None
#     try:
#         print(f"[Tool Log] Attempting to get a connection from the pool...")
#         connection = get_connection()
#         if not connection:
#             print("[Tool Log] Failed to get a connection from the pool")
#             return []
        
#         print(f"[Tool Log] Executing query: {query}")
#         cursor = connection.cursor(dictionary=True)
#         cursor.execute(query)
#         results = cursor.fetchall()
#         print(f"[Tool Log] Query executed successfully. Results: {results}")
        
#         return results
#     except Exception as e:
#         print(f"[Tool Log] Error executing query: {str(e)}")
#         return {
#             "error": str(e)
#         }
#     finally:
#         if connection:
#             print("[Tool Log] Closing connection...")
#             connection.close()

# class State(TypedDict):
#     messages: Annotated[List[AnyMessage], add_messages]
#     intent: str | None
#     schema: str | None  
#     chosen_table: str | None
#     sql_query: str | None
#     query_results: List[Dict] | None
#     performance_metrics: Dict[str, str]
#     thread_id: str | None

# def print_performance_metrics(state: Dict) -> None:
#     """Helper function to print performance metrics"""
#     if "performance_metrics" in state:
#         print("\nPerformance Metrics:")
#         for metric, value in state["performance_metrics"].items():
#             print(f"{metric.replace('_', ' ').title()}: {value}")
#         print()

# def detect_intent_node(state: State) -> Dict[str, Any]:
#     """Detect user intent from the last message using LLM"""
#     start_time = time.time()
#     print("NODE: detect_intent_node - RUNNING")
    
#     try:
#         # Get the last HUMAN message (not AI or Tool messages)
#         user_msg = next((msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), "No user message")
        
#         # Optimized LLM prompt for intent classification
#         prompt = f"""CLASSIFY THIS QUERY IN 1 WORD: "{user_msg}"
        
#         [RULES]
#         Reply "query" ONLY if it contains:
#         - Provider/service search terms: find, list, show, search, providers, services, "near me", location names
#         - Specific service types: plan management, support coordination, therapy
#         - Field-specific terms: phone, email, location, profile
        
#         Reply "conversational" for:
#         - Greetings (hi, hello, bye)
#         - NDIS/process questions
#         - General help/FAQ
#         - Bot identity questions
        
#         [EXAMPLES]
#         "Tell me about attendance of Rose" → query
#         "How you doing today" → conversational
#         "What is Her Employee Id" → query
#         "What's your name?" → conversational
        
#         ANSWER:"""
        
#         # Call the LLM for intent classification
#         intent_response = model.generate_content(prompt)
#         intent = intent_response.text.strip().lower()
#         print(f"Intent detected: {intent}")
        
#         elapsed = time.time() - start_time
#         result = {
#             "messages": [AIMessage(content=f"(Debug) Intent detected: {intent}")],
#             "intent": intent,
#             "performance_metrics": {
#                 **state.get("performance_metrics", {}),
#                 "intent_detection_time": f"{elapsed:.2f}s"
#             }
#         }
#         return result
#     except Exception as e:
#         elapsed = time.time() - start_time
#         return {
#             "messages": [AIMessage(content="Sorry, I had trouble understanding your intent.")],
#             "intent": None,
#             "performance_metrics": {
#                 **state.get("performance_metrics", {}),
#                 "intent_detection_time": f"{elapsed:.2f}s"
#             }
#         }

# def get_schema_node(state: State) -> Dict[str, List[ToolMessage]]:
#     """Get schemas for relevant tables based on user query"""
#     start_time = time.time()
#     print("\nNODE: get_schema_node - RUNNING")
    
#     try:
#         # Get schema for the tables we need
#         # relevant_tables = ["users_data", "list_professions", "users_reviews"]
        
#         # In get_schema_node
#         relevant_tables = ['clients', 'projects', 'chapters', 'tasks', 'users', 'roles']
#         schemas = {}
#         for table in relevant_tables:
#             schemas[table] = get_schema_tool(table)
#         state["schema"] = "\n\n".join([f"-- {table}\n{schema}" for table, schema in schemas.items()])

#         all_schemas = []
        
#         for table in relevant_tables:
#             print(f"[Tool Log] Getting schema for {table}")
#             schema = get_schema_tool.invoke(table)
#             all_schemas.append(schema)
        
#         # Combine all schemas with clear separation
#         combined_schemas = "\n\n=== TABLE SCHEMAS ===\n\n".join(all_schemas)
        
#         # Store in state - This is the key fix!
#         state["schema"] = combined_schemas
#         #print(f"Schema stored in state: {combined_schemas[:200]}...")  
        
#         elapsed = time.time() - start_time
#         return {
#             "messages": [
#                 ToolMessage(
#                     content=f"Retrieved schemas for tables: {', '.join(relevant_tables)}",
#                     tool_name="get_schema_tool",
#                     tool_call_id="schema_lookup",
#                     additional_kwargs={"schemas": combined_schemas}
#                 )
#             ],
#             "schema": combined_schemas,
#             "performance_metrics": {
#                 **state.get("performance_metrics", {}),
#                 "get_schema_time": f"{elapsed:.2f}s"
#             }
#         }
#     except Exception as e:
#         elapsed = time.time() - start_time
#         return {
#             "messages": [
#                 ToolMessage(
#                     content="Failed to retrieve schemas",
#                     tool_name="get_schema_tool",
#                     tool_call_id="schema_lookup"
#                 )
#             ],
#             "schema": None,
#             "performance_metrics": {
#                 **state.get("performance_metrics", {}),
#                 "get_schema_time": f"{elapsed:.2f}s"
#             }
#         }

# def should_continue_after_intent(state: State) -> str:
#     """
#     Decide next node based on recognized intent.
#     """
#     print("NODE: should_continue_after_intent - RUNNING")
#     intent = state.get("intent")
#     print(f"Current intent: {intent}")
    
#     if intent == "conversational":
#         print("Routing to conversational_agent_node")
#         return "conversational_agent_node"
#     elif intent == "query":
#         print("Routing to get_schema_node")
#         return "get_schema_node"
#     else:
#         print("No recognized intent, defaulting to conversational_agent_node")
#         return "conversational_agent_node"

# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# def call_model_with_retry(messages):
#     print("Retrying to get the response")
#     return model.generate_content(messages)

# def conversational_agent_node(state: State) -> Dict[str, List[AIMessage]]:
#     start_time = time.time()
#     print("NODE: conversational_agent_node - RUNNING")
    
#     try:
#         # Get all messages from state
#         messages = state["messages"]
        
#         # Convert messages to Gemini format, filtering out SQL-related messages
#         conversation_history = []
#         for msg in messages:
#             # Skip debug messages, SQL queries, and query results
#             if (isinstance(msg, AIMessage) and isinstance(msg.content, str) and 
#                 (msg.content.startswith("(Debug)") or 
#                  msg.content.startswith("Generated SQL:") or 
#                  msg.content.startswith("Retrieved schemas"))):
#                 continue
#             if isinstance(msg, ToolMessage):
#                 continue
                
#             # Convert to Gemini format
#             conversation_history.append({
#                 "parts": [{"text": msg.content}],
#                 "role": "user" if isinstance(msg, HumanMessage) else "model"
#             })
        
#         # Add system message at the beginning
#         conversation_history.insert(0, {
#             "parts": [{"text": """You are an intelligent chatbot for a Book Publishing Management System. 
#             Maintain context of the conversation and remember important details.
#             If asked about previous questions or information, recall them from the conversation history.
#             """}],
#             "role": "user"
#         })

#         # Generate response with proper error handling
#         try:
#             response = model.generate_content(conversation_history)
#             content = response.text
#         except Exception as e:
#             print(f"Gemini API error: {str(e)}")
#             content = "Sorry, I'm having trouble generating a response right now."

#         return {
#             "messages": [AIMessage(content=content)],
#             "performance_metrics": {
#                 **state.get("performance_metrics", {}),
#                 "conversational_response_time": f"{(time.time()-start_time):.2f}s"
#             }
#         }
        
#     except Exception as e:
#         print(f"Conversational Error: {str(e)}")
#         return {
#             "messages": [AIMessage(content="Error in conversation")],
#             "performance_metrics": {
#                 **state.get("performance_metrics", {}),
#                 "error": str(e)[:100]  # Truncate long errors
#             }
#         }

# def refine_query_node(state: State) -> Dict[str, Any]:
#     """Refine the user query into a SQL query"""
#     start_time = time.time()
#     print("NODE: refine_query_node - RUNNING")
#     try:
#         user_msg = next((msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), "")
        
#         combined_prompt = f"""As an SQL expert, convert this user request into a valid SQL query using EXACT field names from these schemas:

# USER REQUEST: "{user_msg}"

# SCHEMAS:
# {state["schema"]}
# """
        
#         response = model.generate_content(combined_prompt)
#         sql_query = response.text.strip()
#         sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
#         print(f"[Refine Query Node] Generated SQL: {sql_query}")
#         elapsed = time.time() - start_time
#         return {
#             "messages": [AIMessage(content=f"Generated SQL: {sql_query}")],
#             "sql_query": sql_query,
#             "performance_metrics": {
#                 **state.get("performance_metrics", {}),
#                 "refine_query_time": f"{elapsed:.2f}s"
#             }
#         }
#     except Exception as e:
#         elapsed = time.time() - start_time
#         print(f"[Refine Query Node] Error: {str(e)}")
#         return {
#             "messages": [AIMessage(content="Sorry, I had trouble processing your query.")],
#             "performance_metrics": {
#                 **state.get("performance_metrics", {}),
#                 "refine_query_time": "N/A"
#             }
#         }

# def execute_query_node(state: State) -> Dict[str, Any]:
#     """Execute the SQL query using the tool and return results"""
#     start_time = time.time()
#     print("NODE: execute_query_node - RUNNING")
    
#     sql_query = state.get("sql_query")
#     if not sql_query:
#         elapsed = time.time() - start_time
#         return {
#             "messages": [ToolMessage(
#                 content="No SQL query found to execute.",
#                 tool_name="execute_query_tool",
#                 tool_call_id="query_execution"
#             )],
#             "performance_metrics": {
#                 **state.get("performance_metrics", {}),
#                 "execute_query_time": f"{elapsed:.2f}s"
#             }
#         }
    
#     try:
#         # Use the tool to execute the query
#         results = execute_query_tool.invoke(sql_query)
        
#         elapsed = time.time() - start_time
#         result = {
#             "messages": [ToolMessage(
#                 content=results,
#                 tool_name="execute_query_tool",
#                 tool_call_id="query_execution"
#             )],
#             "query_results": results,
#             "performance_metrics": {
#                 **state.get("performance_metrics", {}),
#                 "execute_query_time": f"{elapsed:.2f}s"
#             }
#         }
#         print_performance_metrics(result)
#         return result
        
#     except Exception as e:
#         elapsed = time.time() - start_time
#         return {
#             "messages": [ToolMessage(
#                 content=f"Error executing query: {str(e)}",
#                 tool_name="execute_query_tool",
#                 tool_call_id="query_execution"
#             )],
#             "performance_metrics": {
#                 **state.get("performance_metrics", {}),
#                 "execute_query_time": f"{elapsed:.2f}s"
#             }
#         }

# # def explain_results_node(state: State) -> Dict[str, Any]:
# #     """Generate a natural language explanation of the SQL results with formatted HTML"""
# #     start_time = time.time()
# #     print("NODE: explain_results_node - RUNNING")
    
# #     query_results = state.get("query_results", [])
# #     if not query_results:
# #         return {
# #             "messages": [AIMessage(content="No results found.")]
# #         }
    
# #     # Generate the explanation with proper HTML structure
# #     explanation = ['<div class="providers-list">']
# #     explanation.append('    <h1>Here are the providers I found:</h1>')
# #     explanation.append('    ')  # Blank line after header
    
# #     for idx, result in enumerate(query_results):
# #         # Clean up location components and create full location string
# #         location = result.get('address1', '')
# #         if location:
# #             if result.get('city'):
# #                 location += f", {result.get('city')}"
# #             if result.get('state_code'):
# #                 location += f", {result.get('state_code')}"
        
# #     #     # Format provider details with proper indentation
# #     #     provider_html = f"""    <div class="provider">
# #     #     <h2>{result.get('company', 'Unknown')}</h2>
# #     #     <div class="provider-details">
# #     #         <p>
# #     #             <strong>Phone:</strong> {result.get('phone_number', 'N/A')}<br>
# #     #             <strong>Email:</strong> {result.get('email', 'N/A')}<br>
# #     #             <strong>Location:</strong> {location or 'N/A'}<br>
# #     #             <strong>Profile:</strong> <a href="https://mitrat.com.au/{result['filename']}" target="_blank">View Profile</a>
# #     #         </p>
# #     #     </div>
# #     # </div>"""
        
# #         # explanation.append(provider_html)
        
# #     #     # Add spacing between providers except after the last one
# #     #     if idx < len(query_results) - 1:
# #     #         explanation.append('    ')
    
# #     # explanation.append('</div>')
    
# #     # Calculate elapsed time and prepare return object
# #     elapsed = time.time() - start_time
# #     result = {
# #         "messages": [AIMessage(content="\n".join(explanation))],
# #         "performance_metrics": {
# #             **state.get("performance_metrics", {}),
# #             "explain_results_time": f"{elapsed:.2f}s"
# #         }
# #     }
# #     print_performance_metrics(result)
# #     return result

# def explain_results_node(state):
#     # Get the last tool message which contains the database results
#     messages = state.get("messages", [])
#     for message in reversed(messages):
#         if isinstance(message, ToolMessage):
#             results = message.content
#             break
    
#     # If no results or empty results
#     if not results or (isinstance(results, list) and len(results) == 0):
#         response = "I couldn't find any data matching your query."
#         return {"messages": add_messages(state["messages"], [AIMessage(content=response)])}
    
#     # Transform the raw data into conversational format
#     if isinstance(results, list):
#         if len(results) == 1:
#             # Single result formatting
#             item = results[0]
#             response = "Here's what I found: "
#             for key, value in item.items():
#                 readable_key = key.replace("_", " ").title()
#                 response += f"\n- {readable_key}: {value}"
#         else:
#             # Multiple results formatting
#             response = f"I found {len(results)} records. Here's a summary:\n\n"
#             for i, item in enumerate(results, 1):
#                 if 'employee_name' in item:
#                     response += f"{i}. {item['employee_name']}"
#                     if 'department' in item:
#                         response += f" works in the {item['department']} department"
#                     if 'company' in item:
#                         response += f" at {item['company']}"
#                     response += ".\n"
                    
#                     # Add additional details as needed
#                     details = []
#                     if 'designation' in item:
#                         details.append(f"Designation: {item['designation']}")
#                     if 'email' in item:
#                         details.append(f"Email: {item['email']}")
#                     if 'phone' in item:
#                         details.append(f"Phone: {item['phone']}")
                    
#                     if details:
#                         response += "   " + ", ".join(details) + "\n"
#     else:
#         # Handle non-list results
#         response = f"Query result: {results}"
    
#     return {"messages": add_messages(state["messages"], [AIMessage(content=response)])}

# workflow = None

# def create_workflow():
#     """Creates and returns the workflow with proper visualization"""
#     global workflow
#     if workflow is not None:
#         return workflow
#     workflow_start = time.time()
#     # Create the workflow with MessagesState for memory
#     builder = StateGraph(State)  # Rename to builder for clarity
#     # Add nodes (replace all workflow.add_node with builder.add_node)
#     builder.add_node("detect_intent_node", detect_intent_node)
#     builder.add_node("get_schema_node", get_schema_node)
#     builder.add_node("conversational_agent_node", conversational_agent_node)
#     builder.add_node("refine_query_node", refine_query_node)
#     builder.add_node("execute_query_node", execute_query_node)
#     builder.add_node("explain_results_node", explain_results_node)
#     # Add edges
#     builder.add_edge(START, "detect_intent_node")
#     builder.add_conditional_edges(
#         "detect_intent_node",
#         should_continue_after_intent,
#         {
#             "conversational_agent_node": "conversational_agent_node",
#             "get_schema_node": "get_schema_node",
#         }
#     )
#     builder.add_edge("conversational_agent_node", END)
#     builder.add_edge("get_schema_node", "refine_query_node")
#     builder.add_edge("refine_query_node", "execute_query_node")
#     builder.add_edge("execute_query_node", "explain_results_node")
#     builder.add_edge("explain_results_node", END)
#     # Compile the workflow
#     workflow = builder.compile()
#     elapsed = time.time() - workflow_start
#     print(f"\nPerformance Metrics:")
#     print(f"Total workflow creation time: {elapsed:.2f}s")
#     return workflow

# def main():
#     print("Starting script...")
#     print("Environment variables loaded")
    
#     # Initialize the workflow
#     workflow = create_workflow()
    
#     while True:
#         # Get user input
#         user_query = input("\nEnter your query (or type 'exit' to quit): ")
        
#         if user_query.lower() == 'exit':
#             print("Exiting...")
#             break
            
#         if not user_query.strip():
#             print("Please enter a valid query.")
#             continue
            
#         print(f"\nProcessing query: {user_query}")
        
#         # Start timing for the entire query
#         start_time = time.time()
        
#         # Run the workflow with the user's query
#         result = workflow.invoke({"messages": [HumanMessage(content=user_query)]})
        
#         # Calculate total time
#         total_time = time.time() - start_time
        
#         # Print the final messages
#         print("\nFinal messages:")
#         for message in result["messages"]:
#             if isinstance(message, AIMessage):
#                 print(f"ai: {message.content}")
#             elif isinstance(message, ToolMessage):
#                 print(f"tool: -->{message.content}")
#             elif isinstance(message, HumanMessage):
#                 print(f"human: {message.content}")
        
#         # Print performance metrics
#         print(f"\nTotal time for query: {total_time:.2f}s")
        
#         # Print individual performance metrics if available
#         if "performance_metrics" in result:
#             print("Performance Metrics:")
#             for metric, value in result["performance_metrics"].items():
#                 print(f"{metric}: {value}")

# if __name__ == "__main__":
#     main()

# # Initialize the workflow
# # workflow = create_workflow()

# # from langchain_core.messages import HumanMessage, AIMessage
# # from fastapi import FastAPI, HTTPException
# # from fastapi.middleware.cors import CORSMiddleware
# # from pydantic import BaseModel
# # from typing import Optional, Dict, Any
# # import time
# # import uvicorn
# # import sys
# # import logging

# # # Set up logging
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)

# # # Initialize FastAPI
# # app = FastAPI(
# #     title="HR Management Bot API",
# #     description="API for HR Management Bot with natural language query capabilities",
# #     version="1.0.0"
# # )

# # # Configure CORS (restrict origins in production)
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],   # Update for production
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # # Pydantic models
# # class QueryRequest(BaseModel):
# #     query: str
# #     session_id: Optional[str] = None

# # class QueryResponse(BaseModel):
# #     result: Any
# #     performance_metrics: Dict[str, Any]
# #     session_id: str

# # @app.get("/")
# # async def root():
# #     return {"message": "HR Management Bot API is running"}

# # @app.post("/query", response_model=QueryResponse)
# # async def process_query(request: QueryRequest):
# #     start_time = time.time()
# #     try:
# #         # Generate or use session ID
# #         session_id = request.session_id or f"session_{int(time.time())}"
# #         logger.info(f"Processing query: {request.query} (Session: {session_id})")

# #         # Create HumanMessage for the workflow
# #         human_message = HumanMessage(content=request.query)

# #         # Execute the workflow with proper input
# #         result = workflow.invoke({
# #             "messages": [human_message],
# #             "session_id": session_id,
# #             "performance_metrics": {},  # Initialize performance metrics
# #             "intent": None,
# #             "schema": None,
# #             "chosen_table": None,
# #             "sql_query": None,
# #             "query_results": None,
# #             "thread_id": session_id
# #         })

# #         # Extract the response from the result
# #         response_content = None
# #         for msg in reversed(result.get("messages", [])):
# #             if isinstance(msg, AIMessage) and not msg.content.startswith("(Debug)"):
# #                 response_content = msg.content
# #                 break

# #         # If no AI message found, provide a fallback
# #         if not response_content:
# #             response_content = "No response generated. Please try rephrasing your query."

# #         # Get performance metrics
# #         performance_metrics = result.get("performance_metrics", {})
# #         performance_metrics["total_api_time"] = f"{(time.time() - start_time):.2f}s"

# #         return QueryResponse(
# #             result=response_content,
# #             performance_metrics=performance_metrics,
# #             session_id=session_id
# #         )

# #     except Exception as e:
# #         logger.error(f"Error processing query: {str(e)}", exc_info=True)
# #         raise HTTPException(
# #             status_code=500,
# #             detail=f"Error processing query: {str(e)}"
# #         )

# # if __name__ == "__main__":
# #     print("Starting FastAPI application...")
# #     uvicorn.run("HR_management_bot:app", host="127.0.0.1", port=8000, reload=True)