# This file have all the utils kind functions
import re
from datetime import datetime, timedelta

def cleaned_sql(sql):
    
    '''
    Cleans the raw SQL generated from LLM (it contains markdown so we gonna clean it)
    '''

    match = re.search(r'```sql\n(.*?)\n```', sql, re.DOTALL)

    if match:
        return match.group(1).strip()
    
    else:
        return ""
    

def is_safe_sql(sql : str) -> bool:
    
    """
    Checks whether the SQL query user provided is safe to execute.
    Blocks potentially dangerous operations like DELETE, DROP, TRUNCATE, UPDATE.
    """

    forbidden_keywords = ['delete', 'drop', 'truncate', 'update', 'alter']

    lowered = sql.lower()

    return not any(keyword in lowered for keyword in forbidden_keywords)




def parse_vague_time_phrases(text: str) -> dict:

    """
    Detects vague time phrases and converts them to start and end dates.
    Returns: {"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"} or {} if not detected.
    """

    text = text.lower()
    phrases = [
        "last month", "this month", "last year", "this year",
        "this week", "last week", "yesterday", "today"
    ]

    for phrase in phrases:
        if phrase in text:
            now = datetime.now()

            if phrase == "last month":
                start = (now.replace(day=1) - timedelta(days=1)).replace(day=1)
                end = start.replace(day=28) + timedelta(days=4)
                end = end - timedelta(days=end.day)
            elif phrase == "this month":
                start = now.replace(day=1)
                end = now
            elif phrase == "last year":
                start = datetime(now.year - 1, 1, 1)
                end = datetime(now.year - 1, 12, 31)
            elif phrase == "this year":
                start = datetime(now.year, 1, 1)
                end = now
            elif phrase == "this week":
                start = now - timedelta(days=now.weekday())
                end = now
            elif phrase == "last week":
                start = now - timedelta(days=now.weekday() + 7)
                end = start + timedelta(days=6)
            elif phrase == "yesterday":
                start = now - timedelta(days=1)
                end = start
            elif phrase == "today":
                start = now
                end = now

        
            return {
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date": end.strftime("%Y-%m-%d")
            }


    return {}


   