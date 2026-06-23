from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool

from pydantic import BaseModel, Field
from typing import Literal 
import os 
from dotenv import load_dotenv
import sqlite3 as sql 


load_dotenv(override=True)


user_input = input("write down your ticket : ")

class TicketCateogories(BaseModel):
    category: Literal["Technical","Billing & Invoice", "General Enquires", "Product Return"] = Field(description="The category of the ticket, which can be one of the following: Technical Support, Billing & Invoice, General Enquires, Product Return.")
    priority: Literal["high","Medium","Low"]= Field(description="The priority of the ticket, which can be one of the following: high, Medium, Low.")
    is_escalted:  bool = Field(description="Whether the ticket is escalated or not, which can be either Yes or No.")
    summary: str=Field(description="A brief summary of the ticket, which should be a concise and clear description of the issue or request.")


@tool
def get_category():
    """ retrive the categories from this class and based on the category assign the ticket"""
    cls = TicketCateogories()
    return [str(cl) for cl in cls.category]

ticket_prompt = ChatPromptTemplate.from_messages([
    "system",(
        "You are an elite automated IT and customer support triage agent. "
        "Analyze the incoming customer support ticket carefully. "
        "Extract the core issue, categorize it accurately, assess the realistic priority, "
        "and determine if immediate escalation is necessary based on sentiment and business impact."
        "based on the emotion and problem , assign appopriate ticket category"

    ),
    "human",("Incoming Ticket : \n{user_input}")
])

ticket_template_invoke = ticket_prompt.invoke({
    "user_input":user_input
})
ticket_agent = create_agent(model="groq:llama-3.3-70b-versatile",
                            response_format=TicketCateogories,
                            tools=[get_category])

response = ticket_agent.invoke(ticket_template_invoke)

print("Reponse from AI", response["messages"][-1].content)