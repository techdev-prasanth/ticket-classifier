from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain.agents.middleware import PIIMiddleware, HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from typing import Literal 
import os 
from dotenv import load_dotenv
import sqlite3 as sql 
from langgraph.types import Command


load_dotenv(override=True)


@tool
def customer_lookup(query):
    """ find the customer details """
    return f"customer record found for {query}"

@tool
def delete_customer_data(table: str):
    """ find and delete the user data from tabkle """
    return f"customer data deleted {table}"


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



def ai_ticket_classifier(user_input: str):

    
    ticket_prompt = ChatPromptTemplate.from_messages([
        "system",(
    "You are an enterprise-grade AI Customer Support Ticket Classification Agent.\n\n"

    "Your ONLY responsibility is to analyze customer support tickets and classify them.\n"
    "You must NEVER answer general knowledge questions, engage in casual conversation, "
    "provide explanations, write code, or perform tasks unrelated to customer support tickets.\n\n"

    "IN-SCOPE REQUESTS:\n"
    "- Customer complaints\n"
    "- Product issues\n"
    "- Technical support requests\n"
    "- Billing and payment problems\n"
    "- Shipping and delivery issues\n"
    "- Account access problems\n"
    "- Returns, refunds, exchanges\n"
    "- Feedback and suggestions\n"
    "- Order-related concerns\n\n"

    "OUT-OF-SCOPE REQUESTS:\n"
    "- General knowledge questions\n"
    "- Geography questions (e.g., 'What is India?')\n"
    "- Programming questions\n"
    "- Mathematics questions\n"
    "- Personal advice\n"
    "- Politics, religion, entertainment, or unrelated topics\n\n"

    "If the user input is NOT a customer support ticket, do NOT classify it.\n"
    "Instead return:\n"
    "summary='Invalid Input: The provided text is not a customer support ticket.'\n"
    "category='Other'\n"
    "priority='Low'\n"
    "is_escalated=False\n\n"

    "For valid support tickets:\n"
    "1. Identify the main issue.\n"
    "2. Assign the most accurate category.\n"
    "3. Determine the realistic business priority.\n"
    "4. Determine whether escalation is required.\n"
    "5. Generate a concise professional summary.\n\n"

    "Priority Rules:\n"
    "- High: Payment failures, security issues, account lockouts, severe customer complaints, critical outages.\n"
    "- Medium: Delivery delays, product defects, refund requests, recurring technical issues.\n"
    "- Low: General inquiries, feedback, product information requests.\n\n"

    "Escalation Rules:\n"
    "- Escalate when there is financial impact, security risk, legal concern, VIP customer impact, or severe negative sentiment.\n"
    "- Otherwise do not escalate.\n\n"

    "Always return a structured classification response."
),
        "human",("Incoming Ticket : \n{user_input}")
    ])

    ticket_template_invoke = ticket_prompt.invoke({
        "user_input":user_input
    })
    ticket_agent = create_agent(model="groq:llama-3.3-70b-versatile",
                                response_format=TicketCateogories,
                                tools=[customer_lookup,
                                       delete_customer_data
                                       ],
                                middleware=[
                                    PIIMiddleware("email",strategy="redact",apply_to_input=True),
                                    PIIMiddleware("credit_card",strategy="redact",apply_to_input=True),

                                    HumanInTheLoopMiddleware(
                                         interrupt_on={
                                              "delete_customer_data":True,
                                              "customer_lookup":False

                                         },
                                         
                                    )
                                      ],
                                    checkpointer=InMemorySaver()
                                )
    
    config = {"configurable":{"thread_id":"session_001"}}
    response = ticket_agent.invoke(ticket_template_invoke,
                                   config=config,
                                   )
    

    response = ticket_agent.invoke(
        Command(
            resume={
                "decisions": [
                    {"type": "approve"}
                ]
            }
        ),
        config=config
    )
    print()
    print()
    print()

    for i in response["messages"]:
        print(i)
    print()
    print()

    return response["structured_response"]