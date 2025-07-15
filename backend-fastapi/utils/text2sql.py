"""
Text2SQL implementation for natural language to SQL conversion.
"""

from typing import List, Dict
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class Text2SQLGenerator:
    def __init__(self):
        self.schema = """
CREATE TABLE outlets (
    id INTEGER PRIMARY KEY,
    name TEXT,
    address TEXT,
    opening_time TEXT,
    closing_time TEXT,
    services TEXT  -- JSON string array
);
"""
        # Initialize Groq
        self.llm = ChatGroq(
            temperature=0,  # Use 0 for more deterministic SQL generation
            model="llama3-8b-8192"
        )
        
        # Create prompt template for SQL generation
        self.sql_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a SQL expert. Generate only valid SQLite queries based on the given schema. Do not include any explanations or comments."),
            ("user", """{schema}

Convert this question to SQL: "{query}"

Rules:
1. Use only the tables and columns shown in the schema
2. Return a valid SQLite query
3. For text searches, use LIKE with wildcards
4. For time comparisons, compare as strings
5. For services, use json_extract() and json_each() for SQLite array handling

Example queries:
Q: "Show me outlets in Bangsar"
A: SELECT * FROM outlets WHERE address LIKE '%Bangsar%';

Q: "Which outlets are open after 8pm?"
A: SELECT * FROM outlets WHERE closing_time > '20:00';

Q: "Find outlets in Petaling Jaya"
A: SELECT * FROM outlets WHERE address LIKE '%Petaling Jaya%';

Q: "Which outlets offer dine-in service?"
A: SELECT * FROM outlets WHERE json_extract(services, '$') LIKE '%Dine-in%';

Q: "Show me outlets with delivery service"
A: SELECT * FROM outlets WHERE json_extract(services, '$') LIKE '%delivery%';

Generate SQL for this query: {query}

Return ONLY the SQL query, nothing else.""")
        ])

    def generate_sql(self, query: str) -> str:
        """Convert natural language query to SQL"""
        try:
            # Use LangChain with Groq
            chain = self.sql_prompt | self.llm
            response = chain.invoke({
                "schema": self.schema,
                "query": query
            })
            return response.content.strip()
        except Exception as e:
            # Fallback to basic search if AI fails
            search_term = query.replace("'", "''")  # Basic SQL injection prevention
            return f"SELECT * FROM outlets WHERE name LIKE '%{search_term}%' OR address LIKE '%{search_term}%'"

# Create a global instance
sql_generator = Text2SQLGenerator() 