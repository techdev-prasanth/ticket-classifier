from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

from pydantic import BaseModel, Field
from typing import Literal 
import os 
from dotenv import load_dotenv
import sqlite3 as sql 


load_dotenv(override=True)





def ai_ticket_classifier(user_input: str):

    class TicketCateogories(BaseModel):
        category: Literal["Technical",
            "Billing & Invoice",
            "General Enquires",
            "Product Return",
            "General Enquires", 
            "Technical Support",    
            "Account Access",        
            "Feedback & Suggestions",
            "Shiping & Delivery",
            "Order Issues",
        "Returns & Refunds",
        "Payment Issues",
        "Product Information",
        "Account Management",
        "Promotions & Discounts",
        "Damaged or Defective Product",
        "Wrong or Missing Item",
        "Cancellation Requests",
        "Exchange Requests",
        "Tracking Issues",
        "Customer Complaints",
        "Other"] = Field(description="The category of the ticket, which can be one of the following: Technical Support, Billing & Invoice, General Enquires, Product Return.")
        priority: Literal["high","Medium","Low"]= Field(description="The priority of the ticket, which can be one of the following: high, Medium, Low.")
        is_escalted:  bool = Field(description="Whether the ticket is escalated or not, which can be either Yes or No.")
        summary: str=Field(description="A brief summary of the ticket, which should be a concise and clear description of the issue or request.")

    ticket_prompt = ChatPromptTemplate.from_messages([
        "system",(
            "You are an elite automated IT and customer support triage agent. "
            "Analyze the incoming customer support ticket carefully. "
            "Extract the core issue, categorize it accurately, assess the realistic priority, "
            "and determine if immediate escalation is necessary based on sentiment and business impact."
            "based on the emotion and problem , assign appopriate ticket category"
            "assess the realistic priority, and determine if immediate escalation is necessary."

        ),
        "human",("Incoming Ticket : \n{user_input}")
    ])

    ticket_template_invoke = ticket_prompt.invoke({
        "user_input":user_input
    })
    ticket_agent = create_agent(model="groq:qwen/qwen3.6-27b",
                                response_format=TicketCateogories,
                                )
    response = ticket_agent.invoke(ticket_template_invoke)

    print("Reponse from AI", response["structured_response"])
    return response["structured_response"]